import sqlite3
import math
import pandas as pd
from typing import List, Dict, Any, Iterable, overload
from ... import logging


logger = logging.get_logger(name='sqlite3')

PY2SQL3 = {
    str: 'TEXT',
    int: 'INTEGER',
    float: 'REAL',
    bool: 'INTEGER',
    type(None): 'NULL',
}

SQL32PY = {
    'TEXT': str,
    'INTEGER': int,
    'REAL': float,
    'NULL': type(None),
}

SQL3_TYPES = tuple(PY2SQL3.keys())


def get_header(cursor, table):
    cursor.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cursor.fetchall()]


def get_primary_key(cursor, table):
    cursor.execute(f"PRAGMA table_info({table});")
    table_info = cursor.fetchall()

    primary_key = None
    for column in table_info:
        # 如果列的第五个元素（索引为 5-1=4）为 1，表示该列是主键
        if column[5] == 1:
            primary_key = column[1]  # 获取主键列的列名
    return primary_key


def get_sql_value_str(value, recursive=True):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'NULL'
    elif isinstance(value, bool):
        return '1' if value else '0'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, Iterable) and not isinstance(value, str) and recursive:
        return '(' + ', '.join([get_sql_value_str(v, recursive=False) for v in value]) + ')'
    elif isinstance(value, str) and len(value) > 2 and value[0] == '$' and value[-1] == '$':  # column
        return value[1:-1]
    else:
        return '\'' + str(value).replace("'", "''") + '\''


def get_row_dict(row, header):
    row = {header[i]: row[i] for i in range(len(row))}
    row = {k: None if v == 'NULL' else v for k, v in row.items()}
    return row


class SQL3Table(object):
    def __init__(self, database=None, name=None):
        self.database = database if issubclass(type(database), SQLite3Database) else SQLite3Database(database)
        self.cursor = self.database.cursor
        if name is None:
            tbnames = self.database.get_all_tablenames()
            if len(tbnames) == 1:
                name = tbnames[0]
            else:
                raise ValueError(f"Table name must be specified when database has {'no table' if not tbnames else 'multiple tables'}.")
        elif name not in (tbnames := self.database.get_all_tablenames()):
            self.database.create_table(name, {})
        self.name = name
        self.header = get_header(self.cursor, self.name)
        self.primary_key = get_primary_key(self.cursor, self.name)

    @property
    def info(self):
        self.cursor.execute(f"PRAGMA table_info({self.name})")
        infos = self.cursor.fetchall()
        infos = {info[1]: {'type': SQL32PY.get(info[2], None), 'notnull': bool(info[3]), 'default': info[4], 'primary_key': bool(info[5])} for info in infos}
        return infos

    @property
    def col2type(self):
        return {col: info['type'] for col, info in self.info.items()}

    def add_columns(self, col2type: Dict[str, type]):
        for col, coltype in col2type.items():
            if col not in self.header:
                self.cursor.execute(f"ALTER TABLE {self.name} ADD COLUMN {col} {PY2SQL3.get(coltype, 'TEXT')};")
        self.header = get_header(self.cursor, self.name)

    def add_primary_key(self, name):
        if name not in self.header:
            self.cursor.execute(f"ALTER TABLE {self.name} ADD COLUMN {name} INTEGER PRIMARY KEY;")
            self.header = get_header(self.cursor, self.name)
            self.primary_key = name

    def remove_columns(self, columns):
        columns = [col for col in columns if col in self.header]
        self.database.begin_transaction()

        new_table_name = f"{self.name}_tmp"
        new_col2type = {col: coltype for col, coltype in self.col2type.items() if col not in columns}
        self.database.create_table(new_table_name, new_col2type, primary_key=self.primary_key)

        columns_str = ', '.join(new_col2type.keys())
        self.cursor.execute(f"INSERT INTO {new_table_name} ({columns_str}) SELECT {columns_str} FROM {self.name};")
        self.cursor.execute(f"DROP TABLE {self.name};")
        self.cursor.execute(f"ALTER TABLE {new_table_name} RENAME TO {self.name};")
        self.database.commit_transaction()

        self.header = get_header(self.cursor, self.name)
        self.primary_key = get_primary_key(self.cursor, self.name)

    def rename_columns(self, column_mapping):
        column_mapping = {old: new for old, new in column_mapping.items() if old in self.header}

        if not column_mapping:
            return

        self.database.begin_transaction()

        new_table_name = f"{self.name}_tmp"

        new_col2type = {}
        for col, coltype in self.col2type.items():
            if col in column_mapping:
                new_col2type[column_mapping[col]] = coltype
            else:
                new_col2type[col] = coltype

        self.database.create_table(new_table_name, new_col2type, primary_key=self.primary_key)

        old_columns_str = ', '.join(self.header)
        new_columns_str = ', '.join([column_mapping.get(col, col) for col in self.header])

        self.cursor.execute(f"INSERT INTO {new_table_name} ({new_columns_str}) SELECT {old_columns_str} FROM {self.name};")
        self.cursor.execute(f"DROP TABLE {self.name};")
        self.cursor.execute(f"ALTER TABLE {new_table_name} RENAME TO {self.name};")
        self.database.commit_transaction()

        self.header = get_header(self.cursor, self.name)
        self.primary_key = get_primary_key(self.cursor, self.name)

    def select(self, column, statement, distinct=False, **kwargs):
        self.cursor.execute(f"SELECT {'DISTINCT' if distinct else ''}* FROM {self.name} WHERE {column} {statement}")
        return self.cursor.fetchall()

    def select_func(self, func, *args, distinct=False):
        funcname = 'custom_' + func.__name__
        self.database.conn.create_function(funcname, -1, func)
        arg_str = ', '.join([get_sql_value_str(arg) for arg in args])
        self.cursor.execute(f"SELECT {'DISTINCT' if distinct else ''}* FROM {self.name} WHERE {funcname}({arg_str}) = 1")
        return self.cursor.fetchall()

    def select_like(self, column, value, **kwargs):
        return self.select(column, f"LIKE {get_sql_value_str(value)}", **kwargs)

    def select_glob(self, column, value, **kwargs):
        return self.select(column, f"GLOB {get_sql_value_str(value)}", **kwargs)

    def select_between(self, column, lower, upper, **kwargs):
        return self.select(column, f"BETWEEN {get_sql_value_str(lower)} AND {get_sql_value_str(upper)}", **kwargs)

    def select_in(self, column, values, **kwargs):
        return self.select(column, f"IN {get_sql_value_str(values)}", **kwargs)

    def select_not_in(self, column, values, **kwargs):
        return self.select(column, f"NOT IN {get_sql_value_str(values)}", **kwargs)

    def select_is(self, column, value, **kwargs):
        return self.select(column, f"IS {get_sql_value_str(value)}", **kwargs)

    def select_is_not(self, column, value, **kwargs):
        return self.select(column, f"IS NOT {get_sql_value_str(value)}", **kwargs)

    def insert(self, col2data: Dict[str, Any]):
        self.cursor.execute(f"INSERT INTO {self.name} {get_sql_value_str(['$' + key + '$' for key in col2data.keys()])} VALUES {get_sql_value_str(col2data.values())}")

    def insert_or_replace(self, col2data: Dict[str, Any]):
        cmd = f"INSERT OR REPLACE INTO {self.name} {get_sql_value_str(['$' + key + '$' for key in col2data.keys()])} VALUES {get_sql_value_str(col2data.values())}"
        try:
            self.cursor.execute(cmd)
        except sqlite3.OperationalError as e:
            if 'no column' in str(e) in str(e):
                for key, value in col2data.items():
                    if key not in self.header:
                        self.add_columns({key: type(value)})
                self.cursor.execute(cmd)
            else:
                df = pd.DataFrame({col: {'excepted': self.col2type[col], 'received': type(col2data[col])} for col in col2data}).T
                logger.error(f"error occurred when executing command: `{cmd}`")
                logger.error(df, no_prefix=True)
                raise

    def update_where(self, col2data: Dict[str, Any], where: str):
        set_str = ', '.join([f"{k} = {get_sql_value_str(v)}" for k, v in col2data.items()])
        cmd = f"UPDATE {self.name} SET {set_str} WHERE {where}"
        try:
            self.cursor.execute(cmd)
        except sqlite3.OperationalError as e:
            if 'no such column' in str(e):
                for key, value in col2data.items():
                    if key not in self.header:
                        self.add_columns({key: type(value)})
                self.cursor.execute(cmd)

    def sample(self, n=1, randomly=True):
        self.cursor.execute(f"SELECT * FROM {self.name} ORDER BY RANDOM() LIMIT {n}" if randomly else f"SELECT * FROM {self.name} LIMIT {n}")
        return self.cursor.fetchall()

    @overload
    def __getitem__(self, key: str) -> Dict[str, Any]: ...

    @overload
    def __getitem__(self, index: int) -> Dict[str, Any]: ...

    @overload
    def __getitem__(self, range: slice) -> List[Dict[str, Any]]: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            assert self.primary_key is not None, "Primary key must be specified to use string key."
            key = get_sql_value_str(key)
            self.cursor.execute(f"SELECT * FROM {self.name} WHERE {self.primary_key} = {key}")
            item = self.cursor.fetchone()
            if item is None:
                raise KeyError(f"Key {key} not found.")
            return item
        elif isinstance(key, int):
            self.cursor.execute(f"SELECT * FROM {self.name} LIMIT 1 OFFSET {key}")
            item = self.cursor.fetchone()
            if item is None:
                raise IndexError(f"Index {key} out of range.")
            return item
        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                # 如果 stop 是 None，假定要查询所有记录
                self.cursor.execute(f"SELECT * FROM {self.name} LIMIT -1 OFFSET ?", (start,))
            else:
                # 计算需要查询的记录数
                limit = stop - start
                self.cursor.execute(f"SELECT * FROM {self.name} LIMIT ? OFFSET ?", (limit, start))

            rows = self.cursor.fetchall()
            if not rows:
                raise IndexError(f"Index {key} out of range.")
            # 应用 step
            if step is None or step == 1:
                pass
            else:
                rows = [rows[i] for i in range(0, len(rows), step)]
            return rows
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def chunks(self, n):
        r"""
        Split a table into n chunks, with each chunk containing the same number of rows except the last one which may contain fewer rows.
        """
        self.cursor.execute(f"SELECT * FROM {self.name}")
        rows = self.cursor.fetchall()
        size = int(math.ceil(len(rows) / n))
        return [[row for row in rows[i:i + size]] for i in range(0, len(rows), size)]

    def chunk(self, i, n):
        r"""
        Get the i-th chunk of the table.
        """
        self.cursor.execute(f"SELECT * FROM {self.name}")
        rows = self.cursor.fetchall()
        size = int(math.ceil(len(rows) / n))
        return [row for row in rows[i * size:(i + 1) * size]]

    def splits(self, n):
        r"""
        Split a table into some groups of rows, with each group containing n rows except the last one which may contain fewer rows.
        """
        self.cursor.execute(f"SELECT * FROM {self.name}")
        rows = self.cursor.fetchall()
        return [[row for row in rows[i:i + n]] for i in range(0, len(rows), n)]

    def split(self, i, n):
        r"""
        Get the i-th split of the table.
        """
        self.cursor.execute(f"SELECT * FROM {self.name}")
        rows = self.cursor.fetchall()
        return [row for row in rows[i * n:(i + 1) * n]]

    def df(self):
        self.cursor.execute(f"SELECT * FROM {self.name}")
        return pd.DataFrame([row for row in self.cursor.fetchall()], columns=self.header)

    def __str__(self):
        return str(self.df())

    def __repr__(self):
        return str(self.df())


class SQLite3Database(object):
    def __init__(self, fp=None, read_only=False):
        self.fp = fp or ':memory:'
        self.conn = sqlite3.connect(self.fp, check_same_thread=False, uri=read_only and fp != ':memory:')  # auto-commit
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()

    def create_table(self, name, col2type: Dict[str, type] = {}, exists_ok=False, primary_key=None):
        if exists_ok and name in self.get_all_tablenames():
            return self
        if primary_key:
            col2type[primary_key] = col2type.get(primary_key, str)
        assert len(col2type) > 0, "At least one column must be specified."
        col2type_str = ', '.join([f"{k} {PY2SQL3.get(v, 'TEXT')}" for k, v in col2type.items()]) + (f", PRIMARY KEY ({primary_key})" if primary_key else "")
        self.cursor.execute(f"CREATE TABLE {name} ({col2type_str});")
        return self

    def drop_table(self, name):
        self.cursor.execute(f"DROP TABLE {name}")
        return self

    def get_table(self, name, cls=SQL3Table, **kwargs):
        if name not in self.get_all_tablenames():
            raise ValueError(f"Table {name} does not exist. All tables: {self.get_all_tablenames()}")
        assert issubclass(cls, SQL3Table), "cls must be a subclass of SQL3DatabaseTable"
        return cls(self, name, **kwargs)

    def get_all_tablenames(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in self.cursor.fetchall()]

    def get_all_tables(self):
        return [self.get_table(name) for name in self.get_all_tablenames()]

    def begin_transaction(self):
        try:
            self.cursor.execute("BEGIN TRANSACTION;")
        except sqlite3.OperationalError:
            logging.error("Cannot start a transaction within a transaction.")
        return self

    def commit_transaction(self):
        self.cursor.execute("COMMIT;")
        return self

    def rollback(self):
        self.cursor.execute("ROLLBACK;")
        return self

    def vacuum(self):
        self.cursor.execute("VACUUM;")
        return self

    def set_read_only(self):
        self.conn.close()

        if self.fp == ':memory:':
            raise ValueError("Cannot set an in-memory database to read-only mode")
        else:
            self.conn = sqlite3.connect(f'file:{self.fp}?mode=ro', uri=True, check_same_thread=False)  # read-only
            self.cursor = self.conn.cursor()
