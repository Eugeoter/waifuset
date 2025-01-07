import os
import inspect
from typing import Iterable, overload, Any
from index_kits import IndexV2Builder, ArrowIndexV2
from .dataset import Dataset
from .dict_dataset import DictDataset
from .dataset_mixin import FromDiskMixin
from ..data.dict_data import DictData
from ...utils import file_utils
from ... import logging

LOGGER = logging.get_logger('dataset')

INDEX_KITS_DEFAULT_COLUMN_NAMES = ('image', 'md5', 'width', 'height', 'text', 'text_zh')


class IndexKitsData(DictData):
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], dict):
            return args[0]
        return super().__new__(cls)

    def __init__(
        self,
        index_manager: ArrowIndexV2,
        index: int = None,
        **kwargs,
    ):
        self.index_manager = index_manager
        self.index = int(index)
        self.column_names = self.index_manager.get_columns(self.index)
        super().__init__(**kwargs)

    def get(self, key, default=None):
        if key in self.column_names:
            if key == 'image':
                return self.index_manager.get_image(self.index)
            return self.index_manager.get_attribute(self.index, key, default)
        return super().get(key, default)

    def __getattr__(self, name):
        if 'index_manager' in self.__dict__ and name in self.column_names:
            return self.get(name)
        elif name in self.__dict__:
            return self[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        if key in self.column_names:
            return self.get(key)
        elif key in self.__dict__:
            return self[key]
        else:
            return super().__getitem__(key)


class IndexKitsDataset(DictDataset, FromDiskMixin):
    DEFAULT_CONFIG = {
        **Dataset.DEFAULT_CONFIG,
        **FromDiskMixin.DEFAULT_CONFIG,
    }

    def __init__(self, source: logging.Dict[str, Any], **kwargs):
        super().__init__(source, **kwargs)
        self.register_to_config(
            additional_metadata_column=kwargs.get('additional_metadata_column', None),
            primary_key=kwargs.get('primary_key', None),
            fp=kwargs.get('fp', None),
        )

    @property
    def headers(self):
        return list(self.data[next(iter(self.data))].keys()) + list(INDEX_KITS_DEFAULT_COLUMN_NAMES)

    @overload
    def from_disk(cls, fp: str, additional_metadata_column: str = None, primary_key: str = None, **kwargs) -> 'IndexKitsDataset': ...

    @classmethod
    def from_disk(cls, fp, **kwargs):
        if os.path.isdir(fp):
            return cls.from_dir(fp, **kwargs)
        elif os.path.isfile(fp):
            if fp.endswith('.arrow'):
                return cls.from_arrow_file(fp, **kwargs)
            elif fp.endswith('.json'):
                return cls.from_index_file(fp, **kwargs)
            else:
                raise ValueError(f'Unsupported file type: {fp}')
        else:
            raise FileNotFoundError(fp)

    @classmethod
    def from_index_manager(cls, index_manager: ArrowIndexV2, additional_metadata_column=None, primary_key=None, **kwargs):
        data = {}
        primary_key = None if primary_key not in index_manager.get_columns(0) else primary_key
        for ind in LOGGER.tqdm(index_manager.indices, desc='Pre-loading index kits data', level=False):
            additional_metadata = (index_manager.get_attribute(ind, additional_metadata_column) or {}) if additional_metadata_column else {}
            ik_data = IndexKitsData(index_manager, ind, **additional_metadata)
            key = str(ik_data[primary_key] if primary_key else ind)
            data[key] = ik_data
        return cls(data, additional_metadata_column=additional_metadata_column, primary_key=primary_key, **kwargs)

    @classmethod
    def from_dir(cls, fp, index_file=None, additional_metadata_column=None, primary_key=None, **kwargs):
        LOGGER.info(f'Loading index kits dataset from directory: {logging.yellow(fp)}')
        listdir_param_names = inspect.signature(file_utils.listdir).parameters.keys()
        listdir_kwargs = {
            'exts': '.arrow',
            'return_abspath': True,
            'return_path': True,
        }
        listdir_kwargs.update({
            k: v for k, v in kwargs.items() if k in listdir_param_names
        })
        arrow_files = file_utils.listdir(fp, **listdir_kwargs)
        index_file = index_file or os.path.split(arrow_files[0])[0] + '.ik.json'
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        IndexV2Builder(arrow_files).build(index_file)
        index_manager = ArrowIndexV2(index_file)
        return cls.from_index_manager(index_manager, additional_metadata_column=additional_metadata_column, primary_key=primary_key, **kwargs)

    @classmethod
    def from_arrow_files(cls, files: Iterable[str], index_file=None, additional_metadata_column=None, primary_key=None, **kwargs):
        LOGGER.info(f'Loading index kits dataset from arrow files: {logging.yellow(files)}')
        index_file = index_file or os.path.split(files[0])[0] + '.ik.json'
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        IndexV2Builder(files).build(index_file)
        index_manager = ArrowIndexV2(index_file)
        return cls.from_index_manager(index_manager, additional_metadata_column=additional_metadata_column, primary_key=primary_key, **kwargs)

    @classmethod
    def from_index_file(cls, index_file, additional_metadata_column=None, primary_key=None, **kwargs):
        index_manager = ArrowIndexV2(index_file)
        return cls.from_index_manager(index_manager, additional_metadata_column=additional_metadata_column, primary_key=primary_key, **kwargs)

    @classmethod
    def from_arrow_file(cls, arrow_file, index_file=None, additional_metadata_column=None, primary_key=None, **kwargs):
        LOGGER.info(f'Loading index kits dataset from arrow file: {logging.yellow(arrow_file)}')
        index_file = index_file or os.path.split(arrow_file)[0] + '.ik.json'
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        IndexV2Builder([arrow_file]).build(index_file)
        index_manager = ArrowIndexV2(index_file)
        return cls.from_index_manager(index_manager, additional_metadata_column=additional_metadata_column, primary_key=primary_key, **kwargs)
