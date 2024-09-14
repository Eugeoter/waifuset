from typing import Dict, Any, List, overload
from .dataset import Dataset


class DictDataset(Dataset):
    def __init__(self, source: Dict[str, Any], **kwargs):
        if not isinstance(source, dict):
            raise TypeError(f"source must be a dict, not {type(source)}")

        self.data = source
        super().__init__(**kwargs)

    @overload
    def __getitem__(self, key: str) -> Dict: ...

    @overload
    def __getitem__(self, slice: slice) -> List[Dict]: ...

    @overload
    def __getitem__(self, index: int) -> Dict: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, slice):
            return [self.data[k] for k in list(self.data.keys())[key]]
        elif isinstance(key, int):
            return self.data[list(self.data.keys())[key]]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key].update(value)

    def update(self, other):
        self.data.update(other)

    def clear(self):
        self.data.clear()

    def dict(self):
        return self.data

    @classmethod
    def from_dict(cls, dic: Dict, **kwargs):
        if not isinstance(dic, dict):
            raise TypeError(f"dic must be a dict, not {type(dic)}")
        return cls(dic, **kwargs)

    def __add__(self, other):
        if isinstance(other, DictDataset):
            return self.__class__.from_dict({**self.data, **other.data})
        elif isinstance(other, dict):
            return self.__class__.from_dict({**self.data, **other})

    def __iadd__(self, other):
        if isinstance(other, DictDataset):
            self.data.update(other.data)
        elif isinstance(other, dict):
            self.data.update(other)
        return self
