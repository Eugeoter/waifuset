import torch
import os
from PIL import Image
from typing import List, Union
from .const import WS_REPOS
from ... import logging


def repo2path(model_repo_and_path: str):
    if os.path.isfile(model_repo_and_path):
        model_path = model_repo_and_path
    elif os.path.isdir(model_repo_and_path):
        model_path = os.path.join(model_repo_and_path, "model.pth")
    elif model_repo_and_path in WS_REPOS:
        model_path = model_repo_and_path + '/model.pth'
    else:
        raise ValueError(f"Invalid model_repo_and_path: {model_repo_and_path}")
    return model_path


class WaifuScorer(object):
    def __init__(self, model_path: str = None, cache_dir: str = None, device: str = 'cuda', verbose=False):
        self.verbose = verbose
        self.logger = logging.get_logger(self.__class__.__name__)
        if model_path is None:
            model_path = repo2path(WS_REPOS[0])
            if self.verbose:
                self.logger.print(f"model path not set, switch to default: `{model_path}`")
        if not os.path.isfile(model_path):
            from ...utils.file_utils import download_from_url
            self.logger.info(f"model path not found in local, trying to download from url: `{model_path}`")
            model_path = download_from_url(model_path, cache_dir=cache_dir)

        self.logger.print(f"loading pretrained model from `{logging.stylize(model_path, logging.ANSI.YELLOW, logging.ANSI.UNDERLINE)}`")
        with logging.timer("load model", logger=self.logger):
            self.mlp = load_model(model_path, input_size=768, device=device)
            self.model2, self.preprocess = load_clip_models("ViT-L/14", device=device)
            self.device = self.mlp.device
            self.dtype = self.mlp.dtype
            self.mlp.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image]) -> Union[List[float], float]:
        return self.predict(images)

    @torch.no_grad()
    def predict(self, images: List[Union[Image.Image, torch.Tensor]]) -> Union[List[float], float]:
        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
        bs = len(images)
        if bs == 1 and isinstance(images[0], Image.Image):
            images = images*2  # batch norm
        im_emb_arrs = encode_images([img for img in images if isinstance(img, Image.Image)], self.model2, self.preprocess, device=self.device).to(device=self.device, dtype=self.dtype)
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                im_emb_arrs.insert(i, img)
        scores = self.inference(im_emb_arrs)
        if bs == 1:
            scores = scores[0]
        return scores

    @torch.no_grad()
    def inference(self, im_emb_arrs: torch.Tensor) -> float:
        im_emb_arrs = im_emb_arrs.to(device=self.device, dtype=self.dtype)
        predictions = self.mlp(im_emb_arrs)
        scores = predictions.clamp(0, 10).cpu().numpy().reshape(-1).tolist()
        return scores


def load_clip_models(name: str = "ViT-L/14", device='cuda'):
    import clip
    model2, preprocess = clip.load(name, device=device)  # RN50x64
    return model2, preprocess


def load_model(model_path: str = None, input_size=768, device: str = 'cuda', dtype=None):
    from .mlp import MLP
    model = MLP(input_size=input_size)
    if model_path:
        s = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(s)
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
    if isinstance(images, Image.Image):
        images = [images]
    image_tensors = [preprocess(img).unsqueeze(0) for img in images]
    image_batch = torch.cat(image_tensors).to(device)
    image_features = model2.encode_image(image_batch)
    im_emb_arr = normalized(image_features).cpu().float()
    return im_emb_arr
