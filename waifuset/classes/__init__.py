LAZY_IMPORT = True

if LAZY_IMPORT:
    import sys
    from ..utils.module_utils import _LazyModule
    _import_structure = {
        'data.data': ['Data'],
        'data.dict_data': ['DictData'],
        'data.eugedata': ['EugeData'],
        'data.image_info': ['ImageInfo'],
        'data.caption.caption': ['Caption'],
        'database.sqlite3_database': ['SQLite3Database', 'SQL3Table'],
        'dataset.dataset': ['Dataset'],
        'dataset.dataset_group': ['DatasetGroup'],
        'dataset.dict_dataset': ['DictDataset'],
        'dataset.directory_dataset': ['DirectoryDataset'],
        'dataset.parasite_dataset': ['ParasiteDataset'],
        'dataset.json_dataset': ['JSONDataset'],
        'dataset.csv_dataset': ['CSVDataset'],
        'dataset.sqlite3_dataset': ['SQLite3Dataset'],
        'dataset.hakubooru': ['Hakubooru'],
        'dataset.auto_dataset': ['AutoDataset'],
        'data.caption': ['tagging'],
    }
    sys.modules[__name__] = _LazyModule(__name__, globals()['__file__'], import_structure=_import_structure, module_spec=__spec__)

else:
    from .data.data import Data
    from .data.dict_data import DictData
    from .data.eugedata import EugeData
    from .data.image_info import ImageInfo
    from .data.caption.caption import Caption
    from .database.sqlite3_database import SQLite3Database, SQL3Table
    from .dataset.dataset import Dataset
    from .dataset.dataset_group import DatasetGroup
    from .dataset.dict_dataset import DictDataset
    from .dataset.directory_dataset import DirectoryDataset
    from .dataset.parasite_dataset import ParasiteDataset
    from .dataset.json_dataset import JSONDataset
    from .dataset.csv_dataset import CSVDataset
    from .dataset.sqlite3_dataset import SQLite3Dataset
    from .dataset.hakubooru import Hakubooru
    from .dataset.auto_dataset import AutoDataset
    from .data.caption import tagging
