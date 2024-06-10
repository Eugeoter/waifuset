from typing import List, Literal
from .image_info import ImageInfo
from .data_utils import read_attrs
from ...utils import log_utils


class EugeData(ImageInfo):
    def read(self, fp=None, types: List[Literal['txt', 'danbooru']] = None, **kwargs):
        try:
            attrs_dict = read_attrs(fp or self.image_path, types=types, **kwargs)
        except Exception as e:
            log_utils.get_logger(self.__class__.__name__).print(f"failed to read attrs for {self.image_path}: {e}")
            return None
        if not attrs_dict:
            return
        for attr_name, attr_value in attrs_dict.items():
            setattr(self, attr_name, attr_value)
