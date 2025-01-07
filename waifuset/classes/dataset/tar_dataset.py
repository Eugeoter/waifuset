import tarfile
from .dict_dataset import DictDataset
from .dataset_mixin import FromDiskMixin


# TODO
class TarDataset(FromDiskMixin, DictDataset):
    pass
