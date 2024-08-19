LAZY_IMPORT = True

if LAZY_IMPORT:
    import sys
    from .utils.module_utils import _LazyModule
    _import_structure = {
        'classes.data.data': ['Data'],
        'classes.data.dict_data': ['DictData'],
        'classes.data.huggingface_data': ['HuggingFaceData'],
        'classes.data.eugedata': ['EugeData'],
        'classes.data.image_info': ['ImageInfo'],
        'classes.data.caption.caption': ['Caption'],
        'classes.database.sqlite3_database': ['SQLite3Database', 'SQL3Table'],
        'classes.dataset.dataset': ['Dataset'],
        'classes.dataset.dataset_group': ['DatasetGroup'],
        'classes.dataset.dict_dataset': ['DictDataset'],
        'classes.dataset.directory_dataset': ['DirectoryDataset'],
        'classes.dataset.parasite_dataset': ['ParasiteDataset'],
        'classes.dataset.json_dataset': ['JSONDataset'],
        'classes.dataset.csv_dataset': ['CSVDataset'],
        'classes.dataset.sqlite3_dataset': ['SQLite3Dataset'],
        'classes.dataset.hakubooru': ['Hakubooru'],
        'classes.dataset.auto_dataset': ['AutoDataset'],
        'classes.dataset.fast_dataset': ['FastDataset'],
        'classes.dataset.chain_dataset': ['ChainDataset'],
        'components.waifu_tagger': ['WaifuTagger'],
        'components.waifu_scorer': ['WaifuScorer'],
    }
    sys.modules[__name__] = _LazyModule(__name__, globals()['__file__'], import_structure=_import_structure, module_spec=__spec__)

else:
    from .classes.data.data import Data
    from .classes.data.dict_data import DictData
    from .classes.data.huggingface_data import HuggingFaceData
    from .classes.data.eugedata import EugeData
    from .classes.data.image_info import ImageInfo
    from .classes.data.caption.caption import Caption
    from .classes.database.sqlite3_database import SQLite3Database, SQL3Table
    from .classes.dataset.dataset import Dataset
    from .classes.dataset.dataset_group import DatasetGroup
    from .classes.dataset.dict_dataset import DictDataset
    from .classes.dataset.directory_dataset import DirectoryDataset
    from .classes.dataset.parasite_dataset import ParasiteDataset
    from .classes.dataset.json_dataset import JSONDataset
    from .classes.dataset.csv_dataset import CSVDataset
    from .classes.dataset.sqlite3_dataset import SQLite3Dataset
    from .classes.dataset.hakubooru import Hakubooru
    from .classes.dataset.auto_dataset import AutoDataset
    from .classes.dataset.fast_dataset import FastDataset
    from .classes.dataset.chain_dataset import ChainDataset
    from .components.waifu_tagger import WaifuTagger
    from .components.waifu_scorer import WaifuScorer
