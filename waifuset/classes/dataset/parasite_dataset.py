from itertools import islice
from typing import Dict, List, Union, Iterable, Callable, Generator, overload
from .dataset import Dataset
from .dict_dataset import DictDataset
from ... import logging


def get_root(dataset: Dataset) -> Dataset:
    r"""
    Recursively get the root dataset of a dataset.
    """
    p = dataset
    # find the host of host until it is None
    while hasattr(p, 'host') and p.host is not None:
        p = p.host
    return p


class ParasiteDataset(Dataset):
    DEFAULT_CONFIG = {
        **Dataset.DEFAULT_CONFIG,
        'host': None,
        'root': None,
    }

    host: Dataset
    root: Dataset
    part: Dataset

    def __init__(self, source: Dataset, host: Dataset, **kwargs):
        if not isinstance(source, Dataset):
            raise TypeError(f"source must be a Dataset, not {type(source)}")
        if not isinstance(host, Dataset):
            raise TypeError(f"host must be a Dataset, not {type(host)}")

        self.host = host
        self.root = get_root(host)
        self.part = source
        self.register_to_config(
            host=self.host,
            root=self.root,
        )
        super().__init__(**kwargs)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.host, name)
        else:
            return super().__getattr__(name)

    def get_root(self):
        return self.root

    def get_host(self):
        return self.host

    def get_key(self, key: Union[str, int, slice]) -> Union[str, Iterable[str]]:
        r"""
        Convert the key to a valid key in the host dataset.
        """
        if isinstance(key, str):
            return key
        elif isinstance(key, int):
            # use islice to get the key at a specific position
            try:
                key_iter = islice(self.part.keys(), key, key + 1)
                element = next(key_iter)
                return element
            except StopIteration:
                raise IndexError('Index out of range')
        elif isinstance(key, slice):
            if len(self.part) == 0:
                return []
            # process the start, stop and step of the slice
            start = key.start if key.start is not None else 0
            stop = key.stop
            step = key.step if key.step is not None else 1

            if step == 0:
                raise ValueError('slice step cannot be zero')

            if stop is None:
                # cannot slice infinitely on a generator
                raise ValueError('Slice stop cannot be None when slicing a generator')

            # use islice to get the keys in the required range
            key_iter = islice(self.part.keys(), start, stop, step)
            return key_iter
        else:
            raise TypeError(f"key must be a str, int or slice, not {type(key)}")

    @overload
    def __getitem__(self, key: str) -> Dict: ...

    @overload
    def __getitem__(self, index: int) -> Dict: ...

    @overload
    def __getitem__(self, slice: slice) -> List[Dict]: ...

    def __getitem__(self, key):
        key = self.get_key(key)
        if isinstance(key, str):
            return self.root[key]
        else:
            return [self.root[k] for k in key]

    # def __getitem__(self, key):
    #     if isinstance(key, str):
    #         if key in self.part:
    #             return self.root[key]
    #         else:
    #             raise KeyError
    #     elif isinstance(key, int):
    #         key = list(self.part.keys())[key]
    #         return self.root[key]
    #     elif isinstance(key, slice):
    #         keys = list(self.part.keys())[key]
    #         return [self.root[k] for k in keys]
    #     else:
    #         raise KeyError

    def __setitem__(self, key, value):
        if key in self.part:
            self.root[key] = value
        else:
            raise KeyError

    def __delitem__(self, key):
        if key in self.part:
            del self.root[key]
        else:
            raise KeyError

    def __contains__(self, key):
        return key in self.part

    def __iter__(self):
        for key in self.part:
            yield self.root[key]

    def __len__(self):
        return len(self.part)

    def items(self):
        for key in self.part:
            yield key, self.root[key]

    def keys(self):
        for key in self.part:
            yield key

    def values(self):
        for key in self.part:
            yield self.root[key]

    def kvalues(self, key, **kwargs):
        return [item[1] for item in self.kitems(key, **kwargs)]

    def kitems(self, key, **kwargs):
        return [item for item in self.root.kitems(key, **kwargs) if item[0] in self.part]

    def get(self, key, default=None):
        if key in self.part:
            return self.root[key]
        else:
            return default

    def set(self, key, value):
        if key in self.part:
            self.root.set(key, value)
        else:
            raise KeyError

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def clear(self):
        for key in self.part:
            del self.root[key]

    def dict(self):
        return dict(self.items())

    @classmethod
    def from_dict(cls, dic: Dict, host, **kwargs):
        r"""
        Initialize a ParasiteDataset from a dictionary. The keys of the dictionary will be used as the parasite keys.

        Inherit the config from the dataset.
        """
        return cls.from_dataset(DictDataset(dic), host, **kwargs)

    @classmethod
    def from_keys(cls, keys: Iterable[str], host, **kwargs):
        r"""
        Initialize a ParasiteDataset from a list of keys in a host dataset.

        Inherit the config from the dataset.
        """
        return cls.from_dataset(DictDataset({k: None for k in keys}), host, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: Dataset, host=None, **kwargs):
        r"""
        Initialize a ParasiteDataset from a Dataset. If host is not specified, it will be set to the dataset itself.

        Inherit the config from the dataset.
        """
        host = host or dataset
        kwargs = {**host.config, **kwargs}
        kwargs['host'] = host  # overwrite the original host
        return cls(dataset, **kwargs)

    def subset(self, condition: Callable[[Dict], bool], **kwargs):
        kwargs = {'host': self.root, 'type': None, **kwargs}
        return super().subset(condition, **kwargs)

    def sample(self, n=1, randomly=True, **kwargs):
        kwargs = {'host': self.root, 'type': None, **kwargs}
        return super().sample(n, randomly, **kwargs)

    def chunk(self, i, n, **kwargs):
        kwargs = {'host': self.root, 'type': None, **kwargs}
        return super().chunk(i, n, **kwargs)

    def chunks(self, n, **kwargs):
        kwargs = {'host': self.root, 'type': None, **kwargs}
        return super().chunks(n, **kwargs)

    def split(self, i, n, **kwargs):
        kwargs = {'host': self.root, 'type': None, **kwargs}
        return super().split(i, n, **kwargs)

    def splits(self, n, **kwargs):
        kwargs = {'host': self.root, 'type': None, **kwargs}
        return super().splits(n, **kwargs)
