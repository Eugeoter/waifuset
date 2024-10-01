import os
from pathlib import Path
from .dataset import Dataset
from .dataset_mixin import FromDiskMixin


def get_dataset_cls_from_source(source):
    if issubclass(type(source), Dataset):
        return type(source)
    elif isinstance(source, dict):
        from .dict_dataset import DictDataset
        return DictDataset
    elif isinstance(source, (str, Path)):
        ext = os.path.splitext(source)[1]
        if not ext and os.path.isdir(source):
            from .directory_dataset import DirectoryDataset
            return DirectoryDataset
        elif ext == '.sqlite3' or ext == '.db':
            from .sqlite3_dataset import SQLite3Dataset
            return SQLite3Dataset
        elif ext == '.csv':
            from .csv_dataset import CSVDataset
            return CSVDataset
        elif ext == '.json':
            from .json_dataset import JSONDataset
            return JSONDataset
        else:
            raise NotImplementedError
    elif source is None:
        return None


class AutoDataset(object):
    def __new__(cls, source, dataset_cls=None, **kwargs) -> Dataset:
        ds_cls = get_dataset_cls_from_source(source)
        if issubclass(ds_cls, FromDiskMixin):
            ds = ds_cls.from_disk(source, **kwargs)
        else:
            ds = ds_cls(source, **kwargs)
        if dataset_cls is not None:
            ds = dataset_cls.from_dataset(ds, **kwargs)
        return ds

    def __init__(self, source, dataset_cls=None, **kwargs) -> Dataset:
        pass

    @staticmethod
    def dump(dataset, fp, *args, **kwargs):
        from .dataset_mixin import ToDiskMixin
        if isinstance(dataset, Dataset):
            pass
        elif isinstance(dataset, dict):
            from .dict_dataset import DictDataset
            dataset = DictDataset.from_dict(dataset)
        else:
            raise ValueError(f'Unsupported dataset type: {type(dataset)}')
        cls_ = get_dataset_cls_from_source(fp)
        if not issubclass(cls_, ToDiskMixin):
            raise TypeError(f'{cls_} does not support dump')
        dumpset = cls_.from_dataset(dataset, *args, fp=fp, **kwargs)
        dumpset.commit()
