import torch
import clip
import os
import time
from PIL import Image
from typing import List, Union
from .mlp import MLP
from ...utils import log_utils as logu
from ...utils.file_utils import download_from_url

MLP_MODEL_URL = "https://huggingface.co/Eugeoter/waifu-scorer-v2/waifu-scorer-v2-1.pth"
MLP_CACHE_DIR = "./models/laion/"
MLP_MODEL_PATH = "./models/laion/waifu-scorer-v2-1.pth"


class WaifuScorer:
    def __init__(self, model_path: str = MLP_MODEL_PATH, device: str = 'cuda', verbose=False):
        self.verbose = verbose
        if model_path is None:
            if self.verbose:
                print(f"[{logu.blue('waifu-scorer')}] model path not set, switch to default: `{MLP_MODEL_PATH}`")
            model_path = MLP_MODEL_PATH

        if self.verbose:
            tic = time.time()
            logu.info(f"[{logu.blue('waifu-scorer')}] loading pretrained model from `{logu.stylize(model_path, logu.ANSI.YELLOW, logu.ANSI.UNDERLINE)}`")

        if not os.path.isfile(model_path):
            model_path = download_from_url(MLP_MODEL_URL, cache_dir=MLP_CACHE_DIR)

        self.mlp = load_model(model_path, input_size=768, device=device)
        self.model2, self.preprocess = load_clip_models("ViT-L/14", device=device)
        self.device = self.mlp.device
        self.dtype = self.mlp.dtype

        self.mlp.eval()

        if self.verbose:
            toc = time.time()
            print(f"[{logu.blue('waifu-scorer')}] model loaded: time_cost={toc-tic:.2f} | device={self.device} | dtype={self.dtype}")

    @torch.no_grad()
    def __call__(self, images: List[Image.Image]) -> Union[List[float], float]:
        if isinstance(images, Image.Image):
            images = [images]
        n = len(images)
        if n == 1:
            images = images*2  # batch norm
        images = encode_images(images, self.model2, self.preprocess, device=self.device).to(device=self.device, dtype=self.dtype)
        predictions = self.mlp(images)
        scores = predictions.clamp(0, 10).cpu().numpy().reshape(-1).tolist()
        if n == 1:
            scores = scores[0]
        return scores


def load_clip_models(name: str = "ViT-L/14", device='cuda'):
    model2, preprocess = clip.load(name, device=device)  # RN50x64
    return model2, preprocess


def load_model(model_path: str = None, input_size=768, device: str = 'cuda', dtype=None):
    model = MLP(input_size=input_size)
    if model_path:
        s = torch.load(model_path, map_location=device)
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
