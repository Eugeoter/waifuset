from typing import Dict, Any, Literal
from .dict_dataset import DictDataset
from .fast_dataset import FastDataset, accumulate_datasets, load_fast_dataset
from .dataset import Dataset
from ... import logging

logger = logging.get_logger("dataset")


class ChainDataset(Dataset):
    def __init__(self, *source, dataset_cls=None, merge_mode: Literal['union', 'intersection', 'update', 'no'] = 'no', local_only=False, **default_kwargs) -> Dataset:
        super().__init__(**default_kwargs)
        self.merge_mode = merge_mode
        self.datasets = load_fast_dataset(*source, dataset_cls=dataset_cls, merge_mode=self.merge_mode, local_only=local_only, **default_kwargs)
        if not isinstance(self.datasets, list):
            self.datasets = [self.datasets]
        if len(self.datasets) > 1:
            self.headset = DictDataset({})
            self.headset.priority = float('inf')
            self.datasets.append(self.headset)

    def merge(self, mode: Literal['union', 'intersection', 'update'] = 'union'):
        self.merge_mode = mode
        if len(self.datasets) == 1:
            return
        self.datasets.sort(key=lambda x: x.priority if hasattr(x, 'priority') else 0, reverse=True)
        self.datasets = [accumulate_datasets(self.datasets, mode=self.merge_mode, verbose=self.verbose)]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> 'ChainDataset':
        return cls(DictDataset.from_dict(data, **kwargs))

    def __getitem__(self, key):
        item = self.dtype()
        for dataset in self.datasets:
            if key in dataset:
                item.update(dataset[key])
        return item

    def __setitem__(self, key, value):
        return self.headset.__setitem__(key, value)

    def __delitem__(self, key):
        for dataset in self.datasets:
            dataset.__delitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def set(self, key, value):
        return self.__setitem__(key, value)

    def clear(self):
        for dataset in self.datasets:
            dataset.clear()

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def __contains__(self, key):
        return any(key in dataset for dataset in self.datasets)

    def keys(self):
        # iter all dataset.keys() for dataset in self.datasets
        for dataset in self.datasets:
            for key in dataset.keys():
                yield key

    def values(self):
        for dataset in self.datasets:
            for value in dataset.values():
                yield value

    def items(self):
        for dataset in self.datasets:
            for item in dataset.items():
                yield item

    def __len__(self):
        return len(set(self.keys()))

    def __iter__(self):
        for key in self.keys():
            yield key

    def dump(self, fp, mode: Literal['union', 'intersection', 'update'] = 'union', **kwargs):
        self.merge(mode)
        return FastDataset.dump(self.datasets[0], fp, **kwargs)
