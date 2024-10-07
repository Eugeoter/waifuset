import os
import random
import numpy as np
import time
import PIL
from pathlib import Path
from PIL import Image
from typing import List, Callable, Dict, Any, Union, Literal

from .arb import AspectRatioBucketMixin
from ..dict_dataset import DictDataset
from ..fast_dataset import FastDataset
from ....utils import config_utils, image_utils
from .... import logging


class T2IDataset(config_utils.FromConfigMixin, AspectRatioBucketMixin):
    dataset_source: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    max_retries: int = None  # infinite retries
    hf_cache_dir: str = None
    hf_token: str = None

    batch_size: int = 1

    max_dataset_n_workers: int = 1

    image_key_getter: Callable[[dict], str] = lambda img_md, *args, **kwargs: img_md.get('image_key')
    image_getter: Callable[[dict], Image.Image] = lambda self, img_md, *args, **kwargs: img_md.get('image')
    caption_getter: Callable[[dict], str] = lambda self, img_md, *args, **kwargs: img_md.get('tags') or img_md.get('caption') or ''

    allow_crop: bool = True

    def setup(self):
        self.logger = logging.get_logger("dataset")
        self.samplers = self.get_samplers()

        tic = time.time()
        tic_setup = tic
        timer = {}

        self.dataset = self.load_dataset()

        self.logger.print(f"setup dataset: {time.time() - tic_setup:.2f}s")
        for key, value in timer.items():
            self.logger.print(f"    {key}: {value:.2f}s", no_prefix=True)

        self.buckets = self.make_buckets()
        self.batches = self.make_batches()

    def get_t2i_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        sample = dict(
            image_keys=[],
            images=[],
            captions=[],
        )
        for img_key in batch:
            img_md = self.dataset[img_key]
            image = self.get_bucket_image(img_md)
            if image is None:
                raise FileNotFoundError(f"Image and cache not found for `{img_key}`")
            caption = self.get_caption(img_md)

            sample["image_keys"].append(img_key)
            sample["images"].append(image)
            sample["captions"].append(caption)

        return sample

    def get_samplers(self):
        samplers = [sampler for sampler in dir(self) if sampler.startswith('get_') and sampler.endswith('_sample') and callable(getattr(self, sampler))]
        samplers.sort()
        samplers = [getattr(self, sampler) for sampler in samplers]
        samplers.insert(0, samplers.pop(samplers.index(self.get_t2i_sample)))  # prepend t2i_sampler
        return samplers

    def shuffle(self):
        random.shuffle(self.batches)

    def load_dataset(self) -> DictDataset:
        with self.logger.timer("load dataset"):
            self.logger.info(f"loading dataset...")
            from waifuset.classes.dataset.fast_dataset import parse_source_input
            self.dataset_source = parse_source_input(self.dataset_source)
            default_kwargs = dict(
                cache_dir=self.hf_cache_dir,
                token=self.hf_token,
                max_retries=self.max_retries,
                read_only=True,
                verbose=True,
            )
            dataset = FastDataset(self.dataset_source, dataset_cls=DictDataset, merge_mode='union', **default_kwargs)

        self.logger.print(dataset, no_prefix=True)
        return dataset

    def load_data(self, img_md) -> Dict:
        img_md = self.get_preprocessed_img_md(img_md)
        extra_kwargs = {}
        if extra_kwargs:
            img_md.update(**extra_kwargs)

    def parse_source_input(self, source):
        if source is None:
            return []
        if not isinstance(source, list):
            source = [source]
        source = [dict(name_or_path=src) if isinstance(src, (str, Path)) else src for src in source]
        return source

    def make_batches(self) -> List[List[str]]:
        if self.arb:
            assert hasattr(self, 'buckets'), "You must call `make_buckets` before making batches."
            if self.batch_size == 1:
                batches = [[img_key] for img_key in self.dataset.keys()]
            else:
                batches = []
                for img_keys in self.logger.tqdm(self.buckets.values(), desc='make batches'):
                    for i in range(0, len(img_keys), self.batch_size):
                        batch = img_keys[i:i+self.batch_size]
                        batches.append(batch)
        else:
            img_keys = list(self.dataset.keys())
            batches = []
            for i in self.logger.tqdm(range(0, len(self.dataset), self.batch_size), desc='make batches'):
                batch = img_keys[i:i+self.batch_size]
                batches.append(batch)
        return batches

    def get_samples(self, batch: List[str]):
        samples = {}
        for sampler in self.samplers:
            samples.update(sampler(batch, samples))
        return samples

    def __getitem__(self, i):
        batch = self.batches[i]
        sample = self.get_samples(batch)
        return sample

    def __len__(self):
        return len(self.batches)

    def get_size(self, img_md, update=False):
        image_size, original_size, bucket_size = None, None, None
        if (image_size := img_md.get('image_size')) is None:
            if os.path.exists(img_path := img_md.get('image_path', '')):
                image_size = image_utils.get_image_size(img_path)
            elif (image := self.open_image(img_md)) is not None:
                image_size = image.size
            elif (latents_size := self.get_latents_size(img_md)) is not None:
                image_size = (latents_size[0] * 8, latents_size[1] * 8)
                bucket_size = image_size  # precomputed latents
            else:
                self.logger.error(f"Failed to get image size for: {img_md}")
                raise ValueError("Failed to get image size.")
        image_size = image_utils.convert_size_if_needed(image_size)

        if (original_size := img_md.get('original_size')) is None:
            original_size = image_size
        original_size = image_utils.convert_size_if_needed(original_size)

        if bucket_size is None and (bucket_size := img_md.get('bucket_size')) is None:
            bucket_size = self.get_bucket_size(img_md, image_size=image_size)
        bucket_size = image_utils.convert_size_if_needed(bucket_size)

        if update:
            img_md.update(
                image_size=image_size,
                original_size=original_size,
                bucket_size=bucket_size,
            )
        return image_size, original_size, bucket_size

    def verify_image(self, img_md):
        img_path = img_md.get('image_path')
        if img_path is None or not os.path.exists(img_path):
            return False
        with Image.open(img_path) as img:
            img.verify()
        return True

    def open_image(self, img_md) -> Image.Image:
        if (image := self.image_getter(img_md)) is not None:
            assert isinstance(image, Image.Image), f"image must be an instance of PIL.Image.Image, but got {type(image)}: {image}"
        elif (image := img_md.get('image')) is not None:
            assert isinstance(image, Image.Image), f"image must be an instance of PIL.Image.Image, but got {type(image)}: {image}"
        elif os.path.exists(img_path := img_md.get('image_path', '')):
            try:
                image = Image.open(img_path)
            except PIL.Image.DecompressionBombError:
                self.logger.warning(f"DecompressionBombError: {img_md['image_key']}")
                return None
        else:
            self.logger.warning(f"failed to open image for {img_md['image_key']}")
            return None
        return image

    def get_image(self, img_md, type: Literal['pil', 'numpy'] = 'pil') -> Union[Image.Image, np.ndarray]:
        image = self.open_image(img_md)
        if image is None:
            return None
        image = image_utils.convert_to_rgb(image)
        image = image_utils.rotate_image_straight(image)
        if type == 'numpy':
            image = np.array(image, np.uint8)  # (H, W, C)
        return image

    def get_bucket_image(self, img_md, type: Literal['pil', 'numpy'] = 'pil', update=True) -> Union[Image.Image, np.ndarray]:
        image = self.get_image(img_md, type=type)
        if image is None:
            return None
        _, _, bucket_size = self.get_size(img_md, update=update)
        crop_ltrb = self.get_crop_ltrb(img_md, update=update)
        image = image_utils.crop_ltrb_if_needed(image, crop_ltrb)
        image = image_utils.resize_if_needed(image, bucket_size)
        return image

    def get_crop_ltrb(self, img_md, update=True):
        if img_md.get('crop_ltrb') is not None:
            return img_md['crop_ltrb']

        image_size = img_md['image_size']

        if not self.allow_crop:
            return (0, 0, image_size[0], image_size[1])

        bucket_size = img_md['bucket_size']
        max_ar = self.max_aspect_ratio
        img_w, img_h = image_size
        tar_w, tar_h = bucket_size

        ar_image = img_w / img_h
        ar_target = tar_w / tar_h

        if max_ar is not None and image_utils.aspect_ratio_diff(image_size, bucket_size) > max_ar:
            if ar_image < ar_target:
                new_height = img_w / ar_target * max_ar
                new_width = img_w
            else:
                new_width = img_h * ar_target / max_ar
                new_height = img_h

            left = max(0, int((img_w - new_width) / 2))
            top = max(0, int((img_h - new_height) / 2))
            right = int(left + new_width)
            bottom = int(top + new_height)
            crop_ltrb = (left, top, right, bottom)
        else:
            crop_ltrb = (0, 0, img_w, img_h)
        if update:
            img_md['crop_ltrb'] = crop_ltrb
        return crop_ltrb

    def get_caption(self, img_md):
        return self.caption_getter(img_md)

    @staticmethod
    def collate_fn(batch):
        return batch[0]
