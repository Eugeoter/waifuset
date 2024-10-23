import torch
from PIL import Image
from typing import List
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ... import logging

logger = logging.get_logger("aesthetic_shadow")


class AestheticShadow(object):
    def __init__(self, processor, model, verbose=True):
        self.processor = processor
        self.model = model
        self.verbose = verbose

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=None,
        cache_dir=None,
        verbose=True
    ):
        if pretrained_model_name_or_path is None:  # auto
            raise ValueError("pretrained_model_name_or_path should not be None")
        logger.info(f"Loading pretrained model from `{logging.stylize(pretrained_model_name_or_path, logging.ANSI.YELLOW, logging.ANSI.UNDERLINE)}`", disable=not verbose)
        logger.info(f"  - Device: {str(device)}", disable=not verbose)
        processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForImageClassification.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
        model = model.to(device=device, dtype=dtype)
        return cls(processor, model, verbose)

    def __call__(self, images: List[Image.Image]) -> List[float]:
        if not isinstance(images, list):
            images = [images]
        images = [img.convert("RGB") for img in images]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs).logits
        scores = outputs.softmax(dim=1).detach().cpu()[..., 0].float().numpy().tolist()
        return scores
