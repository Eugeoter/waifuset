import sqlite3
import os
from typing import Dict, List, Literal, Callable, Iterable, overload
from .dataset import Dataset
from .dataset_mixin import DiskIOMixin
from ..database.sqlite3_database import SQLite3Database, SQL3Table, get_sql_value_str, get_row_dict
from ..data.data import Data
from ...const import StrPath


class SQLite3Dataset(SQLite3Database, DiskIOMixin, Dataset):
    DEFAULT_CONFIG = {
        **Dataset.DEFAULT_CONFIG,
        **DiskIOMixin.DEFAULT_CONFIG,
        'tbname': None,
        'primary_key': None,
        'col2type': None,
    }

    def __init__(self, source: StrPath = None, tbname=None, **kwargs):
        primary_key = kwargs.pop('primary_key', None)
        col2type = kwargs.pop('col2type', None)
        SQLite3Database.__init__(self, fp=source)  # set self.path here
        self.register_to_config(fp=self.fp)
        if tbname is None:
            self.table: SQL3Table = None
        else:
            if tbname not in self.get_all_tablenames() and tbname is not None and primary_key is not None:
                self.create_table(tbname, primary_key=primary_key, col2type=col2type)
            self.set_table(tbname)
            self.register_to_config(
                tbname=self.table.name,
                primary_key=self.table.primary_key,
                col2type=self.types,
            )
        Dataset.__init__(self, **kwargs)

    @classmethod
    def from_disk(cls, fp, **kwargs):
        return cls(fp, **kwargs)

    @property
    def info(self):
        return self.table.info

    def set_table(self, table_name):
        self.table = self.get_table(table_name)

    @overload
    def __getitem__(self, key: str) -> Dict: ...

    @overload
    def __getitem__(self, slice: slice) -> List[Dict]: ...

    @overload
    def __getitem__(self, index: int) -> Dict: ...

    def __getitem__(self, key):
        item = self.postprocessor(self.table[key])
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

    def postprocessor(self, row, enable=True):
        return get_row_dict(row, self.table.header) if enable else row

    def items(self, postprocess=True):
        self.cursor.execute(f"SELECT * FROM {self.table.name}")
        for row in self.cursor.fetchall():
            val = self.postprocessor(row, enable=postprocess)
            key = val[self.table.primary_key] if postprocess else row[0]
            yield key, val

    def keys(self):
        self.cursor.execute(f"SELECT {self.table.primary_key} FROM {self.table.name}")
        for row in self.cursor.fetchall():
            yield row[0]

    def values(self, postprocess=True):
        self.cursor.execute(f"SELECT * FROM {self.table.name}")
        for row in self.cursor.fetchall():
            val = self.postprocessor(row, enable=postprocess)
            val.pop(self.table.primary_key)
            yield val

    def kvalues(self, key, distinct=False, where: str = None, **kwargs):
        command = f"SELECT {'DISTINCT ' if distinct else ''}{key} FROM {self.table.name}"
        if where is not None:
            command += f" WHERE {where}"
        self.cursor.execute(command)
        for row in self.cursor.fetchall():
            yield row[0]

    def kitems(self, key, where: str = None, **kwargs):
        command = f"SELECT {self.table.primary_key}, {key} FROM {self.table.name}"
        if where is not None:
            command += f" WHERE {where}"
        self.cursor.execute(command)
        for row in self.cursor.fetchall():
            yield row[0], row[1]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def clear(self):
        self.cursor.execute(f"DELETE FROM {self.table.name}")

    def set(self, key, value):
        if key in self:
            self.table.update_where(value, where=f"{self.table.primary_key} = {get_sql_value_str(key)}")
        else:
            self.table.insert_or_replace({self.table.primary_key: key, **value})

    def dump(self, fp, mode='w', algorithm: Literal['auto', 'iterdump', 'backup'] = 'iterdump'):
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

    def apply_map(self, func: Callable[[Dict], Dict], *args, **kwargs):
        self.begin_transaction()
        super().apply_map(func, *args, **kwargs)
        self.commit_transaction()

    @property
    def header(self):
        return self.table.header

    @property
    def types(self):
        return {col: info['type'] for col, info in self.info.items()}

    def df(self):
        return self.table.df()

    def dict(self):
        return dict(self.items(postprocess=True))

    @classmethod
    def from_dict(cls, dic: Dict, **kwargs):
        tbname = kwargs.pop('tbname', None)
        primary_key = kwargs.pop('primary_key', None)
        assert tbname is not None and primary_key is not None, "Both `tbname` and `primary_key` must be provided"
        if dic:
            v0 = next(iter(dic.values()))
            col2type = {k: type(v) for k, v in v0.items()}
        else:
            col2type = {}
        dataset = cls(tbname=tbname, primary_key=primary_key, col2type=col2type, **kwargs)
        for k, v in dic.items():
            v[primary_key] = k
            dataset[k] = v
        return dataset

    @overload
    def subkeys(self, condition: Callable[[Dict], bool], **kwargs) -> Iterable[str]: ...

    @overload
    def subkeys(self, column: str, statement: str, **kwargs) -> Iterable[str]: ...

    def subkeys(self, condition_or_column, statement=None, **kwargs):
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

    def subset(self, condition_or_column, statement=None, **kwargs):
        if isinstance(condition_or_column, str) and statement is not None:
            return self.select(condition_or_column, statement, **kwargs)
        else:
            kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, **kwargs}
            return super().subset(condition_or_column, **kwargs)

    def sample(self, n=1, randomly=True, type=None, **kwargs):
        sample_lst = self.table.sample(n, randomly)
        sample_lst = [self.postprocessor(s) for s in sample_lst]
        sample_dicts = {s[self.table.primary_key]: s for s in sample_lst}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.sample', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(sample_dicts, **kwargs)

    def chunk(self, i, n, type=None, **kwargs):
        chunk_lst = self.table.chunk(i, n)
        chunk_lst = [self.postprocessor(c) for c in chunk_lst]
        chunk_dicts = {c[self.table.primary_key]: c for c in chunk_lst}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.chunk', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(chunk_dicts, **kwargs)

    def chunks(self, n, type=None, **kwargs):
        chunk_lsts = self.table.chunks(n)
        chunk_lsts = [[self.postprocessor(c) for c in chunk_lst] for chunk_lst in chunk_lsts]
        chunk_dicts = [{c[self.table.primary_key]: c for c in chunk_lst} for chunk_lst in chunk_lsts]
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.chunk', 'verbose': self.verbose, **kwargs}
        return [(type or self.__class__).from_dict(chunk_dict, **kwargs) for chunk_dict in chunk_dicts]

    def split(self, i, n, type=None, **kwargs):
        split_lst = self.table.split(i, n)
        split_lst = [self.postprocessor(s) for s in split_lst]
        split_dict = {s[self.table.primary_key]: s for s in split_lst}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.split', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(split_dict, **kwargs)

    def splits(self, n, type=None, **kwargs):
        split_lsts = self.table.splits(n)
        split_lsts = [[self.postprocessor(s) for s in split_lst] for split_lst in split_lsts]
        split_dicts = [{s[self.table.primary_key]: s for s in split_lst} for split_lst in split_lsts]
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.split', 'verbose': self.verbose, **kwargs}
        return [(type or self.__class__).from_dict(split_dict, *kwargs) for split_dict in split_dicts]

    def subset_from_select(self, select_rows, type=None, **kwargs):
        select_rows = [self.postprocessor(row) for row in select_rows]
        select_dict = {row[self.table.primary_key]: row for row in select_rows}
        kwargs = {'tbname': self.table.name, 'primary_key': self.table.primary_key, 'name': self.name + '.subset', 'verbose': self.verbose, **kwargs}
        return (type or self.__class__).from_dict(select_dict, **kwargs)

    def copy(self):
        raise NotImplementedError

    def select(self, column, statement, type=None, **kwargs):
        return self.subset_from_select(self.table.select(column, statement, **kwargs), type=type)

    def select_func(self, func, *args, type=None, **kwargs):
        return self.subset_from_select(self.table.select_func(func, *args, **kwargs), type=type)

    def select_like(self, column, value, type=None, **kwargs):
        return self.subset_from_select(self.table.select_like(column, value, **kwargs), type=type)

    def select_glob(self, column, value, type=None, **kwargs):
        return self.subset_from_select(self.table.select_glob(column, value, **kwargs), type=type)

    def select_between(self, column, lower, upper, type=None, **kwargs):
        return self.subset_from_select(self.table.select_between(column, lower, upper, **kwargs), type=type)

    def select_in(self, column, values, type=None, **kwargs):
        return self.subset_from_select(self.table.select_in(column, values, **kwargs), type=type)

    def select_not_in(self, column, values, type=None, **kwargs):
        return self.subset_from_select(self.table.select_not_in(column, values, **kwargs), type=type)

    def select_is(self, column, value, type=None, **kwargs):
        return self.subset_from_select(self.table.select_is(column, value, **kwargs), type=type)

    def select_is_not(self, column, value, type=None, **kwargs):
        return self.subset_from_select(self.table.select_is_not(column, value, **kwargs), type=type)
