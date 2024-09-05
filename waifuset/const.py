from pathlib import Path
from typing import Union

StrPath = Union[str, Path]
WAIFUSET_ROOT = Path(__file__).parent
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.jfif', '.webp'}
CAPTION_EXT = '.txt'
CACHE_EXT = '.npz'

FILETYPE2EXTS = {
    'image': IMAGE_EXTS,
    'caption': {CAPTION_EXT},
    'cache': {CACHE_EXT},
}
