import os
from .dataset_mixin import FromDiskMixin
from .dict_dataset import DictDataset
from ...utils import file_utils


class DirectoryDataset(DictDataset, FromDiskMixin):
    DEFAULT_CONFIG = {
        **DictDataset.DEFAULT_CONFIG,
        **FromDiskMixin.DEFAULT_CONFIG,
    }

    @classmethod
    def from_disk(cls, fp, fp_key='path', exts=None, recur=True, fp_type=str, fp_abspath=False, **kwargs):
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
