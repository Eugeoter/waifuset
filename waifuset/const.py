from pathlib import Path
from typing import Union

ROOT = Path(__file__).parent

StrPath = Union[str, Path]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.webp'}
CAPTION_EXT = '.txt'
CACHE_EXT = '.npz'

WD_REPOS = ["SmilingWolf/wd-swinv2-tagger-v3", "SmilingWolf/wd-vit-tagger-v3", "SmilingWolf/wd-convnext-tagger-v3"]
WS_REPOS = ["Eugeoter/waifu-scorer-v3"]
