from pathlib import Path
from typing import Union

StrPath = Union[str, Path]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.webp'}
CAPTION_EXT = '.txt'
CACHE_EXT = '.npz'

WD_MODEL_PATH = './models/wd/swinv2.onnx'
WD_LABEL_PATH = './models/wd/selected_tags.csv'
