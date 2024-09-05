import pandas as pd
import copy
import functools
from typing import Callable, Dict, List, overload
from abc import abstractmethod
from collections import OrderedDict
from .dataset_mixin import ConfigMixin
from ... import logging


def get_header(dic):
    columns = OrderedDict()
    for row in dic.values():
        for h in row.keys():
            if h not in columns:
                columns[h] = None
    return list(columns.keys())


def get_column_types(dic, header=None):
    header = header or get_header(dic)
    types = {}
    for v in dic.values():
        for i in range(len(header)):
            h = header[i]
            if h not in types and v.get(h, None) is not None:
                types[h] = type(v[h])
                header.pop(i)
                if len(header) == 0:
                    return types
                break
    return types


class Dataset(ConfigMixin):
    DEFAULT_CONFIG = {
        **ConfigMixin.DEFAULT_CONFIG,
        'name': None,
        'dtype': None,
        'verbose': False,
    }

    def __init__(self, name=None, dtype=None, verbose=False, **kwargs):
        ConfigMixin.__init__(self)
        self.name = name + '.' + self.__class__.__name__ if name else self.__class__.__name__
        self.dtype = dtype or dict
        self.verbose = verbose
        self.logger = logging.get_logger(name=self.name, disable=not self.verbose)
        self.register_to_config(
            name=self.name,
            dtype=self.dtype,
            verbose=self.verbose,
        )

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def __contains__(self, key):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def get(self, key, default=None):
        pass

    @overload
    @abstractmethod
    def set(self, key, value):
        pass

    @overload
    @abstractmethod
    def set(self, dic: Dict):
        pass

    @abstractmethod
    def update(self, other):
        pass

    @abstractmethod
    def clear(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, dic: Dict, **kwargs):
        pass

    @property
    def header(self):
        if not hasattr(self, '_header'):
            self.update_header()
        return self._header

    def update_header(self, header=None):
        self._header = header or get_header(self.dict())

    @functools.cached_property
    def types(self):
        return get_column_types(self.dict(), self.header)

    @classmethod
    def from_dataset(cls, dataset: 'Dataset', **kwargs):
        if dataset.__class__ == cls:
            return dataset
        kwargs = {**dataset.config, **kwargs}
        return cls.from_dict(dataset.dict(), **kwargs)

    def dict(self):
        return dict(self.items())

    def df(self):
        d = self.dict()
        if not d:
            return pd.DataFrame()
        data = d.values()
        return pd.DataFrame(data, columns=self.header)

    def __str__(self):
        df_str = str(self.df())
        width = max(len(line) for line in df_str.split('\n'))
        title = logging.magenta(self.name.center(width))
        info = logging.yellow(f"size: {len(self)}x{len(self.header)}".center(width))
        return '\n'.join([title, info, df_str])

    def __repr__(self):
        return self.__str__()

    def subkeys(self, condition: Callable[[Dict], bool], **kwargs):
        for k, v in self.items():
            if condition(v):
                yield k

    def subset(self, condition: Callable[[Dict], bool], type=None, **kwargs):
        subset_dict = {}
        for k, v in self.logger.tqdm(self.items(), desc=f"subset", total=len(self)):
            if condition(v):
                subset_dict[k] = v
        config = self.config
        config['name'] += '.subset'
        kwargs = {**config, **kwargs}
        return (type or self.__class__).from_dict(subset_dict, **kwargs)

    def sample(self, n=1, randomly=True, type=None, **kwargs):
        if randomly:
            import random
            sample_dict = dict(random.sample(list(self.items()), n))
        else:
            sample_dict = dict(list(self.items())[:n])
        config = self.config.copy()
        config['name'] += '.subset'
        kwargs = {**config, **kwargs}
        return (type or self.__class__).from_dict(sample_dict, **kwargs)

    def chunk(self, i, n, type=None, **kwargs):
        r"""
        Chunk the dataset into some parts with equal size n and get the i-th chunk of the dataset.
        """
        chunk_dict = dict(list(self.items())[i*n:(i+1)*n])
        config = self.config.copy()
        config['name'] += '.chunk'
        kwargs = {**config, **kwargs}
        return (type or self.__class__).from_dict(chunk_dict, **kwargs)

    def chunks(self, n, type=None, **kwargs):
        r"""
        Chunk the dataset into some parts with equal size n and get all chunks of the dataset.
        """
        chunk_dicts = [dict(list(self.items())[i*n:(i+1)*n]) for i in range(n)]
        config = self.config.copy()
        config['name'] += '.chunk'
        kwargs = {**config, **kwargs}
        return [(type or self.__class__).from_dict(chunk_dict, **kwargs) for chunk_dict in chunk_dicts]

    def split(self, i, n, type=None, **kwargs):
        r"""
        Split the dataset into n parts with equal size and get the i-th split of the dataset.
        """
        split_dict = dict(list(self.items())[i::n])
        config = self.config.copy()
        config['name'] += '.split'
        kwargs = {**config, **kwargs}
        return (type or self.__class__).from_dict(split_dict, **kwargs)

    def splits(self, n, type=None, **kwargs):
        r"""
        Split the dataset into n parts with equal size and get all splits of the dataset.
        """
        split_dicts = [dict(list(self.items())[i::n]) for i in range(n)]
        config = self.config.copy()
        config['name'] += '.split'
        kwargs = {**config, **kwargs}
        return [(type or self.__class__).from_dict(split_dict, **kwargs) for split_dict in split_dicts]

    def kvalues(self, key: str, **kwargs):
        for k, kv in self.kitems(key, **kwargs):
            yield kv

    def kitems(self, key: str, **kwargs):
        for k, v in self.items():
            try:
                yield k, v[key]
            except KeyError as e:
                raise KeyError(f"key `{key}` not found in item `{k}`: `{v}`. ") from e

    def redirect(self, columns, tarset: 'Dataset'):
        for k, v in self.logger.tqdm(tarset.items(), desc='redirect'):
            self.set(k, {h: v[h] for h in columns if h in self.header})

    def apply_map(self, func: Callable[[Dict], Dict], *args, condition: Callable[[Dict], bool] = None, **kwargs):
        postfix = {'done': 0, 'skip': 0}
        tqdm_kwargs = dict(total=len(self), desc=func.__name__.replace('_', ' '), postfix=postfix)
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        pbar = self.logger.tqdm(**tqdm_kwargs)
        for k, v in self.items():
            if condition is None or condition(v):
                new_v = func(v, *args, **kwargs)
            else:
                new_v = None
            if new_v is None:
                postfix['skip'] += 1
            else:
                self.set(k, new_v)
                postfix['done'] += 1
            pbar.set_postfix(postfix)
            pbar.update(1)
        self.update_header()

    def with_map(self, func: Callable[[Dict], Dict], *args, condition: Callable[[Dict], bool] = None, **kwargs):
        new = self.copy()
        new.apply_map(func, *args, condition=condition, **kwargs)
        return new

    def add_columns(self, columns, **kwargs):
        tqdm_kwargs = dict(desc='add columns')
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        for k, v in self.logger.tqdm(self.items(), **tqdm_kwargs):
            for col in columns:
                v.setdefault(col, None)
        self.update_header(self.header + [col for col in columns if col not in self.header])
        return self

    def remove_columns(self, columns, **kwargs):
        tqdm_kwargs = dict(desc='remove columns')
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        for k, v in self.logger.tqdm(self.items(), **tqdm_kwargs):
            for col in columns:
                v.pop(col, None)
        self.update_header([col for col in self.header if col not in columns])
        return self

    def rename_columns(self, column_mapping, **kwargs):
        tqdm_kwargs = dict(desc='rename columns')
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        column_mapping = {k: v for k, v in column_mapping.items() if k in self.header and k != v}
        for k, v in self.logger.tqdm(self.items(), **tqdm_kwargs):
            for col, val in v.items():
                if col in column_mapping:
                    val[column_mapping[col]] = val.pop(col)
        self.update_header([column_mapping.get(col, col) for col in self.header])
        return self

    def copy(self):
        return copy.deepcopy(self)

    def batch_keys(self, batch_size) -> List[List[str]]:
        img_keys = list(self.keys())
        return [img_keys[i:i + batch_size] for i in range(0, len(self), batch_size)]

    def set_verbose(self, verbose):
        self.verbose = verbose
        self.logger.get_disable = not verbose
        self.register_to_config(verbose=verbose)

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.register_to_config(dtype=dtype)
