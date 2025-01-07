from pathlib import Path
from .dict_data import DictData
from ...utils import class_utils


class ImageInfo(DictData):
    r"""
    Dev.
    """

    def __init__(self, image_path, **kwargs):
        super().__init__(image_path=image_path, **kwargs)

    def __getattr__(self, key):
        if key in ('key', 'image_key'):
            return getattr(self, 'stem')
        return super().__getattr__(key)

    @class_utils.dict_cached_property
    def image_key(self):
        return Path(self.image_path).stem

    @class_utils.dict_cached_property
    def category(self):
        return Path(self.image_path).parent.name

    @class_utils.dict_cached_property
    def source(self):
        return Path(self.image_path).parent.parent.name
