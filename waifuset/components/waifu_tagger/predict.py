import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union

from ... import logging
from ...const import ROOT
from ..loaders import OnnxModelLoader
from ...utils import image_utils

# WD_CACHE_DIR = os.path.join(ROOT, "models/wd")
WD_REPOS = ["SmilingWolf/wd-swinv2-tagger-v3", "SmilingWolf/wd-vit-tagger-v3", "SmilingWolf/wd-convnext-tagger-v3"]


def repo2path(model_repo_or_path: str):
    if os.path.isfile(model_repo_or_path):
        model_path = model_repo_or_path
        label_path = os.path.join(os.path.dirname(model_path), "selected_tags.csv")
    elif os.path.isdir(model_repo_or_path):
        model_path = os.path.join(model_repo_or_path, "model.onnx")
        label_path = os.path.join(model_repo_or_path, "selected_tags.csv")
    elif model_repo_or_path in WD_REPOS:
        model_path = model_repo_or_path + '/model.onnx'
        label_path = model_repo_or_path + '/selected_tags.csv'
    else:
        raise ValueError(f"Invalid model_repo_or_path: {model_repo_or_path}")
    return model_path, label_path


class WaifuTagger(OnnxModelLoader):
    def __init__(self, model_path=None, label_path=None, cache_dir=None, device='cuda', verbose=False):
        self.verbose = verbose
        self.logger = logging.get_logger(self.__class__.__name__)
        import pandas as pd

        if model_path is None:
            model_path, _ = repo2path(WD_REPOS[0])
            self.logger.print(f"model path not set, switch to default: `{model_path}`")
        else:
            self.logger.print(f"model_path: {model_path}")
        if label_path is None:
            _, label_path = repo2path(WD_REPOS[0])
            self.logger.print(f"label path not set, switch to default: `{label_path}`")
        else:
            self.logger.print(f"label_path: {label_path}")

        # self.logger.print(f"cache_dir: {os.path.abspath(cache_dir)}")

        super().__init__(model_path=model_path, model_url=model_path, cache_dir=cache_dir, device=device, verbose=verbose)

        self.label_path = Path(label_path)
        if not os.path.isfile(self.label_path):
            from ...utils.file_utils import download_from_url
            self.label_path = download_from_url(label_path, cache_dir=cache_dir)

        # Load labels
        df = pd.read_csv(self.label_path)  # Read csv file
        self.tag_names = df["name"].tolist()
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        # Load other parameters
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name
        self.model_target_size = self.model.get_inputs()[0].shape[1]

        if self.verbose:
            self.logger.print(f"label loaded.")

    def __call__(
        self,
        images: Union[List[Image.Image], Image.Image],
        general_threshold: float = 0.35,
        character_threshold: float = 0.35,
    ) -> List[List[str]]:
        import torch

        if not isinstance(images, list):
            images = [images]
        batch_inputs = np.concatenate([self.prepare_image(img) for img in images])

        # Run model
        with torch.no_grad():
            batch_probs = self.model.run(
                [self.label_name],
                {self.input_name: batch_inputs}
            )[0]

        batch_captions = []
        batch_labels = [list(zip(self.tag_names, batch_probs[i].astype(float))) for i in range(len(batch_probs))]

        for labels in batch_labels:
            # Get general and character tags
            general_tags = get_tags(
                labels,
                self.general_indexes,
                general_threshold
            )
            character_tags = get_tags(
                labels,
                self.character_indexes,
                character_threshold
            )

            # Get top general tag
            general_tags = postprocess_tags(general_tags)
            character_tags = postprocess_tags(character_tags)

            tags = character_tags + general_tags
            batch_captions.append(tags)

        return batch_captions

    def prepare_image(self, image: Image.Image):
        target_size = self.model_target_size

        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)


def preprocess_image(image, size: int):
    # 1. load image
    # convert to numpy array
    if isinstance(image, (str, Path)):
        image = image_utils.load_image(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    image = image_utils.cvt2rgb(image)  # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR

    # 2. Pad image with white pixels to (target_size, target_size)
    old_size = image.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    # 3. Resize image accordingly
    if image.shape[0] > size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    elif image.shape[0] < size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

    # 4. Convert image type to fit model required datatype
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    return image


def get_tags(labels, indexes, threshold) -> Dict[str, float]:
    r"""Filter tags whose probability is beyond threshold and convert them to dictionary."""
    names = [labels[i] for i in indexes]
    res = [x for x in names if x[1] > threshold]
    res = dict(res)
    return res


def postprocess_tags(tags: Dict[str, float]) -> List[str]:
    r"""
    Post-process tags by:
        (1) Sort tags by probability
        (2) Transform dict tags to string of labels
        (3) Replace underlines with spaces, parentheses with escape parentheses
    :param tags: Dict of tags mapping from tag name to probability.
    :return: Post-processed string of tags
    """
    tags = dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))
    tags = [tag.replace('_', ' ') for tag in tags.keys()]
    return tags
