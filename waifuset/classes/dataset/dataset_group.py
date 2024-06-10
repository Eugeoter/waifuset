from typing import List
from .dataset import Dataset


class DatasetGroup(Dataset):
    def __init__(self, datasets: List[Dataset], check_keys=True, **kwargs):
        super().__init__(**kwargs)
        self.datasets: List[Dataset] = datasets
        if check_keys:
            keys = set()
            for dataset in self.datasets:
                for key in dataset.keys():
                    if key in keys:
                        raise ValueError(f"Duplicated keys: {key}")
                    keys.add(key)

    def _get_dataset(self, key, raise_error=True):
        for dataset in self.datasets:
            if key in dataset:
                return dataset
        if raise_error:
            raise KeyError(key)
        return None

    def __getitem__(self, key):
        return self._get_dataset(key).__getitem__(key)

    def __setitem__(self, key, value):
        return self._get_dataset(key).__setitem__(key, value)

    def __delitem__(self, key):
        return self._get_dataset(key).__delitem__(key)

    def __contains__(self, key):
        return self._get_dataset(key, raise_error=False) is not None

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        for dataset in self.datasets:
            yield from dataset

    def keys(self):
        for dataset in self.datasets:
            yield from dataset.keys()

    def values(self):
        for dataset in self.datasets:
            yield from dataset.values()

    def items(self):
        for dataset in self.datasets:
            yield from dataset.items()

    def get(self, key, default=None):
        dataset = self._get_dataset(key, raise_error=False)
        return dataset.get(key, default) if dataset is not None else default

    def set(self, key, value):
        return self._get_dataset(key).__setitem__(key, value)

    def update(self, other):
        for key, value in other.items():
            self.set(key, value)

    def kitems(self, key, **kwargs):
        for dataset in self.datasets:
            yield from dataset.kitems(key, **kwargs)

    def kvalues(self, key, **kwargs):
        for dataset in self.datasets:
            yield from dataset.kvalues(key, **kwargs)

    def clear(self):
        for dataset in self.datasets:
            dataset.clear()

    @classmethod
    def from_dict(cls, dic: dict, **kwargs):
        raise NotImplementedError

    def dict(self):
        dic = {}
        for dataset in self.datasets:
            dic.update(dataset.dict())
        return dic
