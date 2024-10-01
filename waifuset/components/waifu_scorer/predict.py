import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from .const import WS_REPOS
from ...utils import image_utils
from ... import logging, const

logger = logging.get_logger("WaifuScorer")


def repo2path(model_repo_and_path: str, use_safetensors=True):
    ext = ".safetensors" if use_safetensors else ".pth"
    if os.path.isfile(model_repo_and_path):
        model_path = model_repo_and_path
    elif os.path.isdir(model_repo_and_path):
        model_path = os.path.join(model_repo_and_path, "model" + ext)
    elif model_repo_and_path in WS_REPOS:
        model_path = model_repo_and_path + '/' + 'model' + ext
    else:
        raise ValueError(f"Invalid model_repo_and_path: {model_repo_and_path}")
    return model_path


class WaifuScorer(object):
    def __init__(
        self,
        mlp: Union[str, None] = None,
        emb_cache_dir: str = None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    ):
        self.logger = logger
        self.emb_cache_dir = emb_cache_dir
        self.verbose = verbose

        self.mlp = mlp
        self.mlp.to(device)
        self.mlp.eval()
        with logging.timer("load components", logger=self.logger):
            self.clip_model, self.clip_preprocessor = load_clip_models("ViT-L/14", device=device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        r"""
        Load WaifuScorer from a pretrained model name or path.

        @param pretrained_model_name_or_path: str, the pretrained model name or path.
        @param cache_dir: str, optional, the cache directory to save the downloaded model.
        @param device: str, optional, the device to run the model, default by auto detect.
        """
        cache_dir = kwargs.pop("cache_dir", None)

        if pretrained_model_name_or_path is None:  # auto
            pretrained_model_name_or_path = WS_REPOS[0]
            logger.warning(f"model path not set, switch to default: `{pretrained_model_name_or_path}`")
        if not os.path.isfile(pretrained_model_name_or_path):
            from huggingface_hub import hf_hub_download
            logger.info(f"downloading pretrained model from `{logging.stylize(pretrained_model_name_or_path, logging.ANSI.YELLOW, logging.ANSI.UNDERLINE)}`")
            pretrained_model_name_or_path = hf_hub_download(
                pretrained_model_name_or_path,
                filename="model.safetensors" if is_safetensors_installed() else "model.pth",
                cache_dir=cache_dir
            )

        logger.info(f"loading pretrained model from `{logging.stylize(pretrained_model_name_or_path, logging.ANSI.YELLOW, logging.ANSI.UNDERLINE)}`")
        mlp = load_model(pretrained_model_name_or_path, input_size=768)
        return cls(mlp, **kwargs)

    @torch.no_grad()
    def __call__(self, inputs: List[Union[Image.Image, torch.Tensor, const.StrPath]], cache_paths: Optional[List[const.StrPath]] = None) -> List[float]:
        return self.predict(inputs, cache_paths)

    @torch.no_grad()
    def predict(self, inputs: List[Union[Image.Image, torch.Tensor, const.StrPath]], cache_paths: Optional[List[const.StrPath]] = None) -> List[float]:
        img_embs = self.encode_inputs(inputs, cache_paths)
        scores = self.inference(img_embs)
        return scores

    @torch.no_grad()
    def inference(self, img_embs: torch.Tensor) -> List[float]:
        img_embs = img_embs.to(device=self.device, dtype=self.dtype)
        predictions = self.mlp(img_embs)
        scores = predictions.clamp(0, 10).cpu().numpy().reshape(-1).tolist()
        return scores

    def to(self, device=None, dtype=None):
        self.mlp.to(device=device, dtype=dtype)
        self.clip_model.to(device=device, dtype=dtype)

    @property
    def device(self):
        return self.mlp.device

    @property
    def dtype(self):
        return self.mlp.dtype

    # def open_image(self, img_path: Union[str, Path]) -> Image.Image:
    #     try:
    #         image = Image.open(img_path)
    #         image.load()
    #     except OSError as e:
    #         self.logger.error(f"error loading image file {img_path} because of {e}, image file deleted.")
    #         try:
    #             os.remove(img_path)
    #         except PermissionError:
    #             self.logger.error(f"error deleting image file {img_path} because of permission denied.")
    #         return None
    #     except FileNotFoundError as e:
    #         self.logger.error(f"error loading image file {img_path} because of {e}, image file not found.")
    #         return None
    #     return image

    def get_image(self, img_path: Union[str, Path]) -> Image.Image:
        image = Image.open(img_path)
        image = image_utils.convert_to_rgb(image)
        image = image_utils.rotate_image_straight(image)
        return image

    def get_cache_path(self, img_path: Union[str, Path]) -> str:
        return os.path.join(self.emb_cache_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.npz')

    def get_cache(self, cache_path: Union[str, Path]) -> torch.Tensor:
        return load_img_emb_from_disk(cache_path, dtype=self.dtype, is_main_process=True, check_nan=False)["emb"]

    def encode_inputs(self, inputs: List[Union[Image.Image, torch.Tensor, const.StrPath]], cache_paths: Optional[List[const.StrPath]] = None) -> torch.Tensor:
        r"""
        Encode inputs to image embeddings. If embedding cache directory is set, it will save the embeddings to disk.
        """
        inputs = inputs.copy()  # copy to avoid inplace modification
        if isinstance(inputs, (Image.Image, torch.Tensor, str, Path)):
            inputs = [inputs]
        if cache_paths is not None:
            if isinstance(cache_paths, (str, Path)):
                cache_paths = [cache_paths]
            assert len(inputs) == len(cache_paths), f"inputs and cache_paths should have the same length, got {len(inputs)} and {len(cache_paths)}"

        # load image embeddings from cache
        if self.emb_cache_dir is not None and os.path.exists(self.emb_cache_dir):
            for i, inp in enumerate(inputs):
                if (cache_paths is not None and os.path.exists(cache_path := cache_paths[i])) or (isinstance(inp, (str, Path)) and os.path.exists(cache_path := self.get_cache_path(inp))):
                    cache = self.get_cache(cache_path)
                    if len(cache.shape) == 1:
                        cache = cache.unsqueeze(0)  # add batch dim
                    inputs[i] = cache  # replace input with cached image embedding (Tensor)

        # open uncached images
        image_or_tensors = [self.get_image(inp) if isinstance(inp, (str, Path)) else inp for inp in inputs]  # e.g. [Tensor, Image, Tensor, Image, Image], same length as inputs
        image_idx = [i for i, img in enumerate(image_or_tensors) if isinstance(img, Image.Image)]  # e.g. [1, 3, 4]
        batch_size = len(image_idx)
        if batch_size > 0:
            images = [image_or_tensors[i] for i in image_idx]  # e.g. [Image, Image, Image]
            if batch_size == 1:
                images = images * 2  # batch norm
            img_embs = encode_images(images, self.clip_model, self.clip_preprocessor, device=self.device)  # e.g. [Tensor, Tensor, Tensor]
            if batch_size == 1:
                img_embs = img_embs[:1]
            # insert image embeddings back to the image_or_tensors list
            for i, idx in enumerate(image_idx):
                image_or_tensors[idx] = img_embs[i].unsqueeze(0)  # add batch dim

            # save image embeddings to cache
        if self.emb_cache_dir is not None:
            os.makedirs(self.emb_cache_dir, exist_ok=True)
            for i, (inp, img_emb) in enumerate(zip(inputs, image_or_tensors)):
                if isinstance(inp, (str, Path)) or cache_paths:
                    cache_path = cache_paths[i] if cache_paths is not None else self.get_cache_path(inp)
                    save_img_emb_to_disk(img_emb, cache_path)
        return torch.cat(image_or_tensors, dim=0)


def is_safetensors_installed():
    try:
        import safetensors
        return True
    except ImportError:
        return False


def load_clip_models(name: str = "ViT-L/14", device='cuda'):
    import clip
    model2, preprocess = clip.load(name, device=device)  # RN50x64
    return model2, preprocess


def load_model(model_path: str = None, input_size=768, device: str = 'cuda', dtype=None):
    from .mlp import MLP
    model = MLP(input_size=input_size)
    if model_path:
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
    if dtype:
        model = model.to(dtype=dtype)
    return model


def normalized(a: torch.Tensor, order=2, dim=-1):
    l2 = a.norm(order, dim, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


@torch.no_grad()
def encode_images(images: List[Image.Image], model2, preprocess, device='cuda') -> torch.Tensor:
    r"""
    Encode images to image embeddings of shape (batch_size, num_features). The input images should be in RGB format.
    """
    if isinstance(images, Image.Image):
        images = [images]
    image_tensors = [preprocess(img).unsqueeze(0) for img in images]
    image_batch = torch.cat(image_tensors).to(device)
    image_features = model2.encode_image(image_batch)
    im_emb_arr = normalized(image_features).cpu().float()
    return im_emb_arr


def open_cache(cache_path, mmap_mode=None, is_main_process=True):
    try:
        cache = np.load(cache_path, mmap_mode=mmap_mode)
        return cache
    except Exception as e:
        if is_main_process:
            import shutil
            backup_path = str(cache_path) + '.bak'
            shutil.move(str(cache_path), backup_path)
            logger.error(f"remove corrupted cache file: {os.path.abspath(cache_path)}, error: {e}")
        return None


def load_img_emb_from_disk(cache_path, dtype=None, mmap_mode=None, is_main_process=True, check_nan=False) -> Dict[str, Any]:
    cache = open_cache(cache_path, mmap_mode=mmap_mode, is_main_process=is_main_process)
    if cache is None:
        return {}
    img_emb = cache["emb"]
    img_emb = torch.FloatTensor(img_emb).to(dtype=dtype)

    img_emb_flipped = cache.get("emb_flipped", None)
    if img_emb_flipped is not None:
        img_emb_flipped = torch.FloatTensor(img_emb_flipped).to(dtype=dtype)

    if check_nan and torch.any(torch.isnan(img_emb)):
        img_emb = torch.where(torch.isnan(img_emb), torch.zeros_like(img_emb), img_emb)
        logger.warning(f"NaN detected in image embedding cache file: {cache_path}")

    return {"emb": img_emb, "emb_flipped": img_emb_flipped}


def save_img_emb_to_disk(img_emb, cache_path, img_emb_flipped=None):
    try:
        extra_kwargs = {}
        if img_emb_flipped is not None:
            extra_kwargs.update(emb_flipped=img_emb_flipped.float().cpu().numpy())
        np.savez(
            cache_path,
            emb=img_emb.float().cpu().numpy(),
            **extra_kwargs,
        )
    except KeyboardInterrupt:
        raise
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Failed to save image embedding to {cache_path}")
