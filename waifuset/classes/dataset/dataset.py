import pandas as pd
import copy
import functools
from typing import Callable, Generator, Tuple, Any, Dict, List, Iterable, overload
from abc import abstractmethod
from collections import OrderedDict
from .dataset_mixin import ConfigMixin
from ... import logging


def get_headers(dic):
    columns = OrderedDict()
    for row in dic.values():
        for h in row.keys():
            if h not in columns:
                columns[h] = None
    return list(columns.keys())


def get_column2type(dic, header=None):
    header = header or get_headers(dic)
    col2type = {}
    for v in dic.values():
        for i in range(len(header)):
            h = header[i]
            if h not in col2type and v.get(h, None) is not None:
                col2type[h] = type(v[h])
                header.pop(i)
                if len(header) == 0:
                    return col2type
                break
    return col2type


class Dataset(ConfigMixin):
    r"""
    To inherit this class, you need to implement the following methods:
    - __getitem__(self, key)
    - __setitem__(self, key, value)
    - __delitem__(self, key)
    - __contains__(self, key)
    - __iter__(self)
    - __len__(self)
    - items(self)
    - keys(self)
    - values(self)
    - get(self, key, default=None)
    - set(self, key, value)
    - set(self, dic: Dict)
    - update(self, other)
    - clear(self)
    - from_dict(cls, dic: Dict, **kwargs)
    """
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
    def headers(self) -> List[str]:
        r"""
        Return the header of the dataset.
        """
        if not hasattr(self, '_header'):
            self.update_header()
        return self._header

    @property
    def column_names(self) -> List[str]:
        r"""
        Return the column names of the dataset. Alias of `header`.
        """
        return self.headers

    def update_header(self, header=None):
        self._header = header or get_headers(self.dict())

    @functools.cached_property
    def types(self) -> Dict[str, type]:
        r"""
        Get the mapping from column name to its data type.
        """
        return get_column2type(self.dict(), self.headers)

    @classmethod
    def from_dataset(cls, dataset: 'Dataset', **kwargs):
        r"""
        Initialize a new dataset from another dataset.

        Inherit the config from the original dataset and update with the new kwargs.
        """
        # if dataset.__class__ == cls:
        #     return dataset
        kwargs = {**dataset.config, **kwargs}
        kwargs.setdefault('_headers', dataset.headers)
        return cls.from_dict(dataset.dict(), **kwargs)

    def dict(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.items())

    def df(self) -> pd.DataFrame:
        r"""
        Return a pandas DataFrame of the dataset.
        """
        d = self.dict()
        if not d:
            return pd.DataFrame()
        data = d.values()
        return pd.DataFrame(data, columns=self.headers)

    def __str__(self):
        r"""
        Return a well-formatted string representation of the dataset.
        """
        df_str = str(self.df())
        width = max(len(line) for line in df_str.split('\n'))
        title = logging.magenta(self.name.center(width))
        info = logging.yellow(f"size: {len(self)}x{len(self.headers)}".center(width))
        return '\n'.join([title, info, df_str])

    def __repr__(self):
        return self.__str__()

    def subkeys(self, condition: Callable[[Dict], bool], **kwargs) -> Generator[str, None, None]:
        r"""
        Return a generator of the keys in the dataset that satisfy the condition.
        """
        for k, v in self.items():
            if condition(v):
                yield k

    def subset(self, condition: Callable[[Dict], bool], dataset_cls: type = None, **kwargs) -> 'Dataset':
        r"""
        Return a subset of the dataset that satisfy the condition.
        """
        subset_dict = {}
        for k, v in self.logger.tqdm(self.items(), desc=f"subset", total=len(self)):
            if condition(v):
                subset_dict[k] = v
        config = self.config
        config['name'] += '.subset'
        kwargs = {**config, **kwargs}
        return (dataset_cls or self.__class__).from_dict(subset_dict, **kwargs)

    def sample(self, n=1, randomly=True, dataset_cls: type = None, **kwargs) -> 'Dataset':
        if randomly:
            import random
            sample_dict = dict(random.sample(list(self.items()), n))
        else:
            sample_dict = dict(list(self.items())[:n])
        config = self.config.copy()
        config['name'] += '.subset'
        kwargs = {**config, **kwargs}
        return (dataset_cls or self.__class__).from_dict(sample_dict, **kwargs)

    def chunk(self, i: int, n: int, dataset_cls: type = None, **kwargs) -> 'Dataset':
        r"""
        Chunk the dataset into some parts with equal size n and get the i-th chunk of the dataset.

        For example, if the dataset has 10 items and n=3, then the dataset will be chunked into 4 parts:
        | chunk idx | keys |
        | --- | --- |
        | 0 | 0, 1, 2 |
        | 1 | 3, 4, 5 |
        | 2 | 6, 7, 8 |
        | 3 | 9 |
        The chunk 0 will be returned if i=0.
        """
        chunk_dict = dict(list(self.items())[i*n:(i+1)*n])
        config = self.config.copy()
        config['name'] += '.chunk'
        kwargs = {**config, **kwargs}
        return (dataset_cls or self.__class__).from_dict(chunk_dict, **kwargs)

    def chunks(self, n: int, dataset_cls: type = None, **kwargs) -> List['Dataset']:
        r"""
        Chunk the dataset into some parts with equal size n and get all chunks of the dataset.

        Equivalent to making batches of the dataset with the given batch size n.

        For example, if the dataset has 10 items and n=3, then the dataset will be chunked into 4 parts:
        | chunk idx | keys |
        | --- | --- |
        | 0 | 0, 1, 2 |
        | 1 | 3, 4, 5 |
        | 2 | 6, 7, 8 |
        | 3 | 9 |
        """
        items = list(self.items())
        chunk_dicts = [dict(items[i*n:(i+1)*n]) for i in range((len(self)+n-1)//n)]
        config = self.config.copy()
        config['name'] += '.chunk'
        kwargs = {**config, **kwargs}
        return [(dataset_cls or self.__class__).from_dict(chunk_dict, **kwargs) for chunk_dict in chunk_dicts]

    def split(self, i: int, n: int, dataset_cls: type = None, **kwargs) -> 'Dataset':
        r"""
        Split the dataset into n parts with equal size and get the i-th split of the dataset.

        For example, if the dataset has 10 items and n=3, then the dataset will be split into 3 parts:
        | split idx | keys |
        | --- | --- |
        | 0 | 0, 3, 6, 9 |
        | 1 | 1, 4, 7 |
        | 2 | 2, 5, 8 |
        The split 0 will be returned if i=0.
        """
        split_dict = dict(list(self.items())[i::n])
        config = self.config.copy()
        config['name'] += '.split'
        kwargs = {**config, **kwargs}
        return (dataset_cls or self.__class__).from_dict(split_dict, **kwargs)

    def splits(self, n: int, dataset_cls: type = None, **kwargs) -> List['Dataset']:
        r"""
        Split the dataset into n parts with equal size and get all splits of the dataset.

        For example, if the dataset has 10 items and n=3, then the dataset will be split into 3 parts:
        | split idx | keys |
        | --- | --- |
        | 0 | 0, 3, 6, 9 |
        | 1 | 1, 4, 7 |
        | 2 | 2, 5, 8 |
        """
        items = list(self.items())
        split_dicts = [dict(items[i::n]) for i in range(n)]
        config = self.config.copy()
        config['name'] += '.split'
        kwargs = {**config, **kwargs}
        return [(dataset_cls or self.__class__).from_dict(split_dict, **kwargs) for split_dict in split_dicts]

    def kvalues(self, column: str, **kwargs) -> Generator[Any, None, None]:
        r"""
        Return a generator of the values of the key in the dataset.
        """
        if not isinstance(column, str):
            raise ValueError(f"column must be a string, not {type(column)}")
        if column not in self.headers:
            raise ValueError(f"key `{column}` not found in the header: {self.headers}")

        for k, kv in self.kitems(column, **kwargs):
            yield kv

    def kitems(self, column: str, **kwargs) -> Generator[Tuple[str, Any], None, None]:
        r"""
        Return a generator of the items of the column in the dataset.
        """
        if not isinstance(column, str):
            raise ValueError(f"column must be a string, not {type(column)}")
        if column not in self.headers:
            raise ValueError(f"key `{column}` not found in the header: {self.headers}")

        for k, v in self.items():
            try:
                yield k, v[column]
            except KeyError as e:
                raise KeyError(f"key `{column}` not found in item `{k}`: `{v}`. ") from e

    def redirect(self, columns: List[str], tarset: 'Dataset') -> None:
        r"""
        Set the columns of the dataset to the corresponding columns of the target dataset.

        Inplace operation.
        """
        if not isinstance(columns, Iterable) or isinstance(columns, str):
            raise ValueError(f"columns must be an iterable of strings, not {type(columns)}")
        for col in columns:
            if not isinstance(col, str):
                raise ValueError(f"all columns must be strings, but got {col} whose type is {type(col)}")
            if col not in tarset.headers:
                raise ValueError(f"column `{col}` not found in the header of the target dataset: {tarset.headers}")

        if len(self) <= len(tarset):
            for k, v in self.logger.tqdm(self.items(), desc='redirect'):
                if k in tarset:
                    tar_v = tarset[k]
                    self.set(k, {col: tar_v[col] for col in columns if col in tarset.headers})
        else:
            for k, tar_v in tarset.items():
                if k in self:
                    self.set(k, {col: tar_v[col] for col in columns if col in tarset.headers})

    def apply_map(self, func: Callable[[Dict], Dict], *args, condition: Callable[[Dict], bool] = None, **kwargs) -> None:
        r"""
        Apply a function to each item in the dataset and update the dataset with the new values.

        The return value of the function should be a dictionary that maps from column name to the new value.

        Inplace operation.

        @param func: the function to apply to each item in the dataset.
        @param args: the positional arguments to pass to the function.
        @param condition: the condition that the item should satisfy to be updated.
        @param kwargs: the keyword arguments to pass to the function. The keyword arguments starting with 'tqdm_' will be passed to the tqdm logger (with the 'tqdm_' prefix removed).
        """
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

    def with_map(self, func: Callable[[Dict], Dict], *args, condition: Callable[[Dict], bool] = None, **kwargs) -> 'Dataset':
        r"""
        Return a new dataset with the function applied to each item in the dataset.

        The return value of the function should be a dictionary that maps from column name to the new value.

        @param func: the function to apply to each item in the dataset.
        @param args: the positional arguments to pass to the function.
        @param condition: the condition that the item should satisfy to be updated.
        @param kwargs: the keyword arguments to pass to the function. The keyword arguments starting with 'tqdm_' will be passed to the tqdm logger (with the 'tqdm_' prefix removed).
        """
        new = self.copy()
        new.apply_map(func, *args, condition=condition, **kwargs)
        return new

    def add_columns(self, columns: List[str], exist_ok=True, **kwargs) -> 'Dataset':
        r"""
        Add columns to the dataset. The new columns will be filled with None.

        Inplace operation.
        """
        if not isinstance(columns, list):
            raise ValueError(f"columns must be a list, not {type(columns)}")
        for col in columns:
            if not isinstance(col, str):
                raise ValueError(f"all columns must be strings, but got {col} whose type is {type(col)}")
            if not exist_ok and col in self.headers:
                raise ValueError(f"column `{col}` already exists in the header: {self.headers}")

        tqdm_kwargs = dict(desc='add columns')
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        for k, v in self.logger.tqdm(self.items(), **tqdm_kwargs):
            for col in columns:
                v.setdefault(col, None)
        self.update_header(self.headers + [col for col in columns if col not in self.headers])
        return self

    def remove_columns(self, columns: List[str], **kwargs) -> 'Dataset':
        r"""
        Remove columns from the dataset.

        Inplace operation.
        """
        if not isinstance(columns, list):
            raise ValueError(f"columns must be a list, not {type(columns)}")
        for col in columns:
            if not isinstance(col, str):
                raise ValueError(f"all columns must be strings, but got {col} whose type is {type(col)}")
            if col not in self.headers:
                raise ValueError(f"column `{col}` not found in the header: {self.headers}")

        tqdm_kwargs = dict(desc='remove columns')
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        for k, v in self.logger.tqdm(self.items(), **tqdm_kwargs):
            for col in columns:
                v.pop(col, None)
        self.update_header([col for col in self.headers if col not in columns])
        return self

    def rename_columns(self, column_mapping: Dict[str, str], **kwargs) -> 'Dataset':
        r"""
        Rename columns in the dataset.

        Inplace operation.
        """
        if not isinstance(column_mapping, dict):
            raise ValueError(f"column_mapping must be a dictionary, not {type(column_mapping)}")
        for key, row in column_mapping.items():
            if not isinstance(key, str) or not isinstance(row, str):
                raise ValueError(f"all keys and values in column_mapping must be strings, but got {key} and {row} whose types are {type(key)} and {type(row)}")
            if key not in self.headers:
                raise ValueError(f"column `{key}` not found in the header: {self.headers}")
            if row in self.headers:
                raise ValueError(f"column `{row}` already exists in the header: {self.headers}")

        tqdm_kwargs = dict(desc='rename columns')
        tqdm_kwargs.update({k[5:]: v for k, v in kwargs.items() if k.startswith('tqdm_')})
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('tqdm_')}
        column_mapping = {k: v for k, v in column_mapping.items() if k in self.headers and k != v}
        for key, row in self.logger.tqdm(self.items(), **tqdm_kwargs):
            for old_col, new_col in column_mapping.items():
                row[new_col] = row.pop(old_col)
        self.update_header([column_mapping.get(col, col) for col in self.headers])
        return self

    def copy(self) -> 'Dataset':
        return copy.deepcopy(self)

    def batch_keys(self, batch_size: int) -> List[List[str]]:
        r"""
        Return a list of batches of keys in the dataset with the given batch size.

        For example, if the dataset has 10 items and the batch size is 3, then the keys will be batched into 4 parts:
        | batch idx | keys |
        | --- | --- |
        | 0 | 0, 1, 2 |
        | 1 | 3, 4, 5 |
        | 2 | 6, 7, 8 |
        | 3 | 9 |
        """
        if not isinstance(batch_size, int):
            raise ValueError(f"batch_size must be an integer, not {type(batch_size)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, not {batch_size}")

        img_keys = list(self.keys())
        return [img_keys[i:i + batch_size] for i in range(0, len(self), batch_size)]

    def set_verbose(self, verbose: bool) -> None:
        r"""
        Set the verbosity of the dataset.
        """
        self.verbose = verbose
        self.logger.set_disable(not verbose)
        self.register_to_config(verbose=verbose)

    def set_dtype(self, dtype: type) -> None:
        self.dtype = dtype
        self.register_to_config(dtype=dtype)
