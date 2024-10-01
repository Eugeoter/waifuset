import sqlite3
import os
from typing import Dict, List, Any, Tuple, Union, Literal, Callable, Iterable, Generator, overload
from .dataset import Dataset, get_column2type
from .dataset_mixin import DiskIOMixin
from ..database.sqlite3_database import SQLite3Database, SQL3Table, get_sql_value_str, get_row_dict, PY2SQL3
from ..data.data import Data
from ...const import StrPath
from ... import logging


class SQLite3Dataset(SQLite3Database, DiskIOMixin, Dataset):
    DEFAULT_CONFIG = {
        **Dataset.DEFAULT_CONFIG,
        **DiskIOMixin.DEFAULT_CONFIG,
        'tbname': None,
        'primary_key': None,
        'col2type': None,
    }

    def __init__(self, source: StrPath = None, tbname=None, read_only=False, **kwargs):
        Dataset.__init__(self, **kwargs)
        primary_key = kwargs.pop('primary_key', None)
        col2type = kwargs.pop('col2type', {})
        SQLite3Database.__init__(self, fp=kwargs.pop('fp', None) or source, read_only=read_only)  # set self.path here
        self.register_to_config(fp=self.fp, read_only=read_only)
        if tbname is None:
            if len(all_table_names := self.get_all_tablenames()) == 1:
                self.logger.warning(f"since table name is not provided when initializing {self.__class__.__name__}, using \'{all_table_names[0]}\' by default.")
                tbname = all_table_names[0]
                self.set_table(tbname)
            else:
                self.logger.warning(f"table is set to None since no table name is provided when initializing {self.__class__.__name__}, available table names: {all_table_names}")
                self.table: SQL3Table = None
        else:
            if tbname not in self.get_all_tablenames() and primary_key is not None:
                self.logger.warning(f"table name {tbname} not found when initializing {self.__class__.__name__}, creating new table with primary key {primary_key} and {len(col2type)} columns.")
                self.create_table(tbname, primary_key=primary_key, col2type=col2type)
            self.set_table(tbname)

        if self.table is not None:
            self.register_to_config(
                tbname=self.table.name,
                primary_key=self.table.primary_key,
                col2type=self.types,
            )

    @classmethod
    def from_disk(cls, fp: str, **kwargs):
        return cls(fp, **kwargs)

    @property
    def info(self) -> Dict[str, Dict[str, Any]]:
        r"""
        Return a dict containing information of the table, with column names as keys and a dict containing column information as values.

        The dict contains the following keys:
        - `type`: the type of the column
        - `notnull`: whether the column is not null
        - `default`: the default value of the column
        - `primary_key`: whether the column is the primary key
        """
        if self.table is None:
            raise ValueError("table is not set")
        return self.table.info

    def set_table(self, table_name: str) -> None:
        r"""
        Set the table of the dataset by name.
        """
        if not isinstance(table_name, str):
            raise TypeError(f"table_name must be a str, not {type(table_name)}")

        self.table = self.get_table(table_name)

    @overload
    def __getitem__(self, key: str) -> Dict: ...

    @overload
    def __getitem__(self, slice: slice) -> List[Dict]: ...

    @overload
    def __getitem__(self, index: int) -> Dict: ...

    def __getitem__(self, key):
        item = self.table[key]
        item = self.postprocessor(item) if not isinstance(key, slice) else [self.postprocessor(i) for i in item]
        return item

    def __setitem__(self, key, value):
        if isinstance(value, Data):
            value = value.dict()
        self.table.insert_or_replace({self.table.primary_key: key, **value})

    def __delitem__(self, key):
        self.cursor.execute(f"DELETE FROM {self.table.name} WHERE {self.table.primary_key} = {get_sql_value_str(key)}")

    def __contains__(self, key):
        self.cursor.execute(f"SELECT * FROM {self.table.name} WHERE {self.table.primary_key} = {get_sql_value_str(key)}")
        return bool(self.cursor.fetchone())

    def __iter__(self):
        return iter(self.keys())

    def __next__(self):
        return next(self.keys())

    def __len__(self):
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table.name}")
        return self.cursor.fetchone()[0]

    def postprocessor(self, row, enable=True) -> Dict[str, Any]:
        r"""
        Postprocess the row fetched from the database to a dictionary, with column names as keys.
        """
        return get_row_dict(row, self.table.header) if enable else row

    def items(self, postprocess=True, sort_by_column=None) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        self.cursor.execute(f"SELECT * FROM {self.table.name}" + (f" ORDER BY {self.sort_by_column}" if sort_by_column else ""))
        for row in self.cursor.fetchall():
            val = self.postprocessor(row, enable=postprocess)
            key = val[self.table.primary_key] if postprocess else row[0]
            yield key, val

    def keys(self, sort_by_column: str = None, reverse=False) -> Generator[str, None, None]:
        order_clause = f" ORDER BY {sort_by_column} {'DESC' if reverse else 'ASC'}" if sort_by_column else ""
        self.cursor.execute(f"SELECT {self.table.primary_key} FROM {self.table.name}{order_clause}")
        for row in self.cursor.fetchall():
            yield row[0]

    def values(self, postprocess=True, sort_by_column: str = None, reverse=False) -> Generator[Dict[str, Any], None, None]:
        order_clause = f" ORDER BY {sort_by_column} {'DESC' if reverse else 'ASC'}" if sort_by_column else ""
        self.cursor.execute(f"SELECT * FROM {self.table.name}{order_clause}")
        for row in self.cursor.fetchall():
            val = self.postprocessor(row, enable=postprocess)
            val.pop(self.table.primary_key)
            yield val

    def kvalues(self, column: str, distinct=False, where: str = None, sort_by_column: str = None, reverse=False, **kwargs) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        if not isinstance(column, str):
            raise ValueError(f"column must be a string, not {type(column)}")
        if column not in self.header:
            raise ValueError(f"key `{column}` not found in the header: {self.header}")
        if where is not None and not isinstance(where, str):
            raise ValueError(f"where must be a string, not {type(where)}")
        if sort_by_column is not None:
            if not isinstance(sort_by_column, str):
                raise ValueError(f"sort_by_column must be a string, not {type(sort_by_column)}")
            if sort_by_column not in self.header:
                raise ValueError(f"sort_by_column `{sort_by_column}` not found in the header: {self.header}")

        order_clause = f" ORDER BY {sort_by_column} {'DESC' if reverse else 'ASC'}" if sort_by_column else ""
        where_clause = f" WHERE {where}" if where is not None else ""
        self.cursor.execute(f"SELECT {'DISTINCT ' if distinct else ''}{column} FROM {self.table.name}{where_clause}{order_clause}")
        for row in self.cursor.fetchall():
            yield row[0]

    def kitems(self, column: str, where: str = None, sort_by_column: str = None, reverse=False, **kwargs) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        if not isinstance(column, str):
            raise ValueError(f"column must be a string, not {type(column)}")
        if column not in self.header:
            raise ValueError(f"key `{column}` not found in the header: {self.header}")
        if where is not None and not isinstance(where, str):
            raise ValueError(f"where must be a string, not {type(where)}")
        if sort_by_column is not None:
            if not isinstance(sort_by_column, str):
                raise ValueError(f"sort_by_column must be a string, not {type(sort_by_column)}")
            if sort_by_column not in self.header:
                raise ValueError(f"sort_by_column `{sort_by_column}` not found in the header: {self.header}")

        order_clause = f" ORDER BY {sort_by_column} {'DESC' if reverse else 'ASC'}" if sort_by_column else ""
        where_clause = f" WHERE {where}" if where is not None else ""
        self.cursor.execute(f"SELECT {self.table.primary_key}, {column} FROM {self.table.name}{where_clause}{order_clause}")
        for row in self.cursor.fetchall():
            yield row[0], row[1]

    def get(self, key, default=None) -> Dict[str, Any]:
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, other: Union[Dataset, Dict[str, Any]], executemany: bool = False) -> None:
        r"""
        Update the dataset with another dataset or a dictionary.
        """
        if executemany and isinstance(other, SQLite3Dataset):
            column_names = other.table.header
            columns = ', '.join(column_names)
            placeholders = ', '.join(['?' for _ in column_names])
            self.cursor.executemany(f"INSERT OR REPLACE INTO {self.table.name} ({columns}) VALUES ({placeholders})", other.table)
        else:
            for key, value in self.logger.tqdm(other.items(), desc=f"update"):
                self[key] = value

    def clear(self) -> None:
        r"""
        Delete all data in the dataset table.

        Inplace operation.
        """
        self.cursor.execute(f"DELETE FROM {self.table.name}")

    def set(self, key: Union[str, Dict[str, Any]], value: Dict[str, Any] = None) -> None:
        r"""
        Set the value of a key.

        If the key is a string, it will be treated as the primary key of the dataset. And the value will be used to update the data with the key.

        If the key is a dictionary, it will be treated as a single data with the primary key in the dict.
        """
        if isinstance(key, str):
            if key in self:
                self.table.update_where(value, where=f"{self.table.primary_key} = {get_sql_value_str(key)}")
            else:
                self.table.insert_or_replace({self.table.primary_key: key, **value})
        elif isinstance(key, dict):  # single data with primary key in the dict
            if self.table.primary_key not in key:
                raise ValueError(f"primary key {self.table.primary_key} not found in the dict: {key}")

            key, value = key[self.table.primary_key], key
            return self.set(key, value)
        else:
            raise ValueError(f"key must be a str or a dict, not {type(key)}")

    def dump(self, fp: str, mode='w', algorithm: Literal['auto', 'iterdump', 'backup'] = 'iterdump') -> None:
        r"""
        Dump the dataset to a file.

        If the file path is the same as the current file path, the dataset will be committed and no further operation will be performed.

        @param fp: the file path to dump the dataset
        @param mode: the mode to open the file, 'w' for write, 'a' for append
        @param algorithm: the algorithm to dump the dataset, 'auto' for automatic selection, 'iterdump' for iterdump, 'backup' for backup
        """
        if self.fp != ':memory:' and os.path.exists(fp) and os.path.samefile(fp, self.fp):
            self.commit()
            return
        if mode == 'w' and os.path.exists(fp):
            os.remove(fp)

        if (algorithm == 'iterdump') or (algorithm == 'auto' and self.fp == ':memory:' and fp != ':memory:'):  # memory to disk
            if mode == 'a' and os.path.exists(fp):
                ds_bak = SQLite3Dataset(fp, tbname=self.table.name, verbose=False)
                assert ds_bak.table.primary_key == self.table.primary_key, f"primary key mismatch: {ds_bak.table.primary_key} != {self.table.primary_key}"
                for k, v in self.items():
                    ds_bak[k] = v
                ds_bak.commit()
            else:
                conn_bak = sqlite3.connect(fp)
                conn_bak.executescript('\r\n'.join(self.conn.iterdump()))
                conn_bak.close()
        elif (algorithm == 'backup') or (algorithm == 'auto'):
            conn_bak = sqlite3.connect(fp)
            with self.logger.tqdm(total=len(self), desc=f"dump {self.name} to {fp}") as pbar:
                def progress_callback(status, remaining, total):
                    pbar.n = total - remaining
                    pbar.refresh()
            self.conn.backup(conn_bak, progress=progress_callback)
            conn_bak.close()

    def apply_map(self, func: Callable[[Dict], Dict], *args, condition: Callable[[Dict], bool] = None, **kwargs) -> None:
        self.begin_transaction()
        super().apply_map(func, *args, condition=condition, **kwargs)
        self.commit_transaction()

    @property
    def header(self) -> List[str]:
        return self.table.header

    @property
    def types(self) -> Dict[str, str]:
        r"""
        Return a dictionary mapping column names to their python types.
        """
        return {col: info['type'] for col, info in self.info.items()}

    def df(self):
        return self.table.df()

    def dict(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.items(postprocess=True))

    @classmethod
    def from_dict(cls, dic: Dict, **kwargs):
        tbname = kwargs.pop('tbname', None)
        primary_key = kwargs.pop('primary_key', None)
        assert tbname is not None and primary_key is not None, f"Initializing a {cls.__name__} from dict requires both 'tbname' and 'primary_key' to be provided."
        if 'col2type' in kwargs:
            col2type = kwargs.pop('col2type')
        elif dic:
            col2type = get_column2type(dic)
        else:
            col2type = {}
        dataset = cls(tbname=tbname, primary_key=primary_key, col2type=col2type, **kwargs)
        for k, v in dic.items():
            v[primary_key] = k
            dataset[k] = v
        return dataset

    @overload
    def subkeys(self, condition: Callable[[Dict], bool], **kwargs) -> List[str]: ...

    @overload
    def subkeys(self, column: str, statement: str, **kwargs) -> List[str]: ...

    def subkeys(self, condition_or_column, statement=None, **kwargs) -> List[str]:
        if isinstance(condition_or_column, str) and statement is not None:
            kwargs = {'postprocess': False, **kwargs}
            return [row[0] for row in self.table.select(condition_or_column, statement, **kwargs)]
        else:
            kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, **kwargs}
            return super().subkeys(condition_or_column, **kwargs)

    @overload
    def subset(self, condition: Callable[[Dict], bool], **kwargs) -> 'SQLite3Dataset': ...

    @overload
    def subset(self, column: str, statement: str, **kwargs) -> 'SQLite3Dataset': ...

    def subset(self, condition_or_column, statement=None, **kwargs) -> 'SQLite3Dataset':
        kwargs['fp'] = None  # ensure in-memory dataset
        if isinstance(condition_or_column, str) and statement is not None:
            return self.select(condition_or_column, statement, **kwargs)
        else:
            kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, **kwargs}
            return super().subset(condition_or_column, **kwargs)

    def sample(self, n=1, randomly=True, type=None, **kwargs) -> 'SQLite3Dataset':
        sample_lst = self.table.sample(n, randomly)
        sample_lst = [self.postprocessor(s) for s in sample_lst]
        sample_dicts = {s[self.table.primary_key]: s for s in sample_lst}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.sample', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(sample_dicts, **kwargs)

    def chunk(self, i, n, type=None, **kwargs) -> 'SQLite3Dataset':
        chunk_lst = self.table.chunk(i, n)
        chunk_lst = [self.postprocessor(c) for c in chunk_lst]
        chunk_dicts = {c[self.table.primary_key]: c for c in chunk_lst}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.chunk', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(chunk_dicts, **kwargs)

    def chunks(self, n, type=None, **kwargs) -> List['SQLite3Dataset']:
        chunk_lsts = self.table.chunks(n)
        chunk_lsts = [[self.postprocessor(c) for c in chunk_lst] for chunk_lst in chunk_lsts]
        chunk_dicts = [{c[self.table.primary_key]: c for c in chunk_lst} for chunk_lst in chunk_lsts]
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.chunk', 'verbose': self.verbose, **kwargs}
        return [(type or self.__class__).from_dict(chunk_dict, **kwargs) for chunk_dict in chunk_dicts]

    def split(self, i, n, type=None, **kwargs) -> 'SQLite3Dataset':
        split_lst = self.table.split(i, n)
        split_lst = [self.postprocessor(s) for s in split_lst]
        split_dict = {s[self.table.primary_key]: s for s in split_lst}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.split', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(split_dict, **kwargs)

    def splits(self, n, type=None, **kwargs) -> List['SQLite3Dataset']:
        split_lsts = self.table.splits(n)
        split_lsts = [[self.postprocessor(s) for s in split_lst] for split_lst in split_lsts]
        split_dicts = [{s[self.table.primary_key]: s for s in split_lst} for split_lst in split_lsts]
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.split', 'verbose': self.verbose, **kwargs}
        return [(type or self.__class__).from_dict(split_dict, *kwargs) for split_dict in split_dicts]

    def subset_from_select(self, select_rows, type=None, **kwargs) -> 'SQLite3Dataset':
        select_rows = [self.postprocessor(row) for row in select_rows]
        select_dict = {row[self.table.primary_key]: row for row in select_rows}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.subset', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(select_dict, **kwargs)

    def copy(self):
        raise NotImplementedError

    def select(self, column, statement, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select(column, statement, **kwargs), type=type)

    def select_func(self, func, *args, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_func(func, *args, **kwargs), type=type)

    def select_like(self, column, value, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_like(column, value, **kwargs), type=type)

    def select_glob(self, column, value, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_glob(column, value, **kwargs), type=type)

    def select_between(self, column, lower, upper, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_between(column, lower, upper, **kwargs), type=type)

    def select_in(self, column, values, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_in(column, values, **kwargs), type=type)

    def select_not_in(self, column, values, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_not_in(column, values, **kwargs), type=type)

    def select_is(self, column, value, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_is(column, value, **kwargs), type=type)

    def select_is_not(self, column, value, type=None, **kwargs) -> 'SQLite3Dataset':
        return self.subset_from_select(self.table.select_is_not(column, value, **kwargs), type=type)

    def add_columns(self, col2type: Dict[str, Any], **kwargs) -> 'SQLite3Dataset':
        if isinstance(col2type, list):
            col2type = {col: 'TEXT' for col in col2type}
        self.table.add_columns(col2type)
        self.register_to_config(col2type=self.types)
        return self

    def remove_columns(self, columns, **kwargs) -> 'SQLite3Dataset':
        if self.table.primary_key in columns:
            raise ValueError(f"Primary key {self.table.primary_key} cannot be removed")
        self.table.remove_columns(columns)
        self.register_to_config(col2type=self.types)
        return self

    def rename_columns(self, column_mapping, **kwargs) -> 'SQLite3Dataset':
        self.table.rename_columns(column_mapping)
        self.register_to_config(col2type=self.types)

        if self.table.primary_key in column_mapping:
            self.register_to_config(primary_key=column_mapping[self.table.primary_key])
        return self

    def set_column_type(self, column: str, new_type: type) -> None:
        r"""
        Set the type of a column.

        Inplace operation.
        """
        old_type = self.table.col2type.get(column, None)
        if new_type == old_type:
            logging.warning(f"Column {column} is already of type {new_type}.")
            return
        new_type = PY2SQL3.get(new_type, 'TEXT')
        tbname = self.table.name
        self.cursor.execute(f"ALTER TABLE {tbname} RENAME COLUMN {column} TO {column}_old;")
        self.cursor.execute(f"ALTER TABLE {tbname} ADD COLUMN {column} {new_type};")
        self.cursor.execute(f"UPDATE {tbname} SET {column} = {column}_old;")
        self.remove_columns([f"{column}_old"])

    def sort(self, column, reverse=False, **kwargs) -> 'SQLite3Dataset':
        r"""
        Sort the dataset by a column.

        Inplace operation.
        """
        self.table.sort(column, reverse)
        return self
