from typing import Dict, List, Iterable, Callable, overload
from .dataset import Dataset


def get_root(dataset: Dataset) -> Dataset:
    p = dataset
    while hasattr(p, 'host') and p.host is not None:
        p = p.host
    return p


class ParasiteDataset(Dataset):
    DEFAULT_CONFIG = {
        **Dataset.DEFAULT_CONFIG,
        'host': None,
        'root': None,
    }

    def __init__(self, source: Iterable[str], host: Dataset, **kwargs):
        assert issubclass(type(host), Dataset), f"host must be a Dataset, not {type(host)}"
        self.host = host
        self.root = get_root(host)
        self.part = {k: None for k in source} if source else {}
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

    @overload
    def __getitem__(self, key: str) -> Dict: ...

    @overload
    def __getitem__(self, index: int) -> Dict: ...

    @overload
    def __getitem__(self, slice: slice) -> List[Dict]: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.part:
                return self.root[key]
            else:
                raise KeyError
        elif isinstance(key, int):
            key = list(self.part.keys())[key]
            return self.root[key]
        elif isinstance(key, slice):
            keys = list(self.part.keys())[key]
            return [self.root[k] for k in keys]
        else:
            raise KeyError

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
        return cls(dic, host, **kwargs)

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
