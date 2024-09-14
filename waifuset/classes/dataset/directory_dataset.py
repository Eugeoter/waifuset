import os
from typing import Iterable
from .dataset_mixin import FromDiskMixin
from .dict_dataset import DictDataset
from ...utils import file_utils


class DirectoryDataset(DictDataset, FromDiskMixin):
    DEFAULT_CONFIG = {
        **DictDataset.DEFAULT_CONFIG,
        **FromDiskMixin.DEFAULT_CONFIG,
    }

    @classmethod
    def from_disk(cls, fp: str, fp_key='path', exts: Iterable[str] = None, recur: bool = True, fp_type: type = str, fp_abspath: bool = False, **kwargs):
        if not isinstance(fp, str):
            raise TypeError(f"fp must be a str, not {type(fp)}")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"fp not found: {fp}")
        if not os.path.isdir(fp):
            raise NotADirectoryError(f"fp must be a directory, not a file: {fp}")

        data = {os.path.splitext(os.path.basename(f))[0]: {fp_key: f} for f in file_utils.listdir(
            fp,
            exts=exts,
            return_type=fp_type,
            return_abspath=fp_abspath,
            return_path=True,
            recur=recur
        )}
        config = dict(fp=fp, exts=exts, recur=recur, fp_type=fp_type, fp_abspath=fp_abspath, **kwargs)
        return cls(data, **config)
