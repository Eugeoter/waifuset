import json
import os
from typing import Dict, Any
from .dict_dataset import DictDataset
from .dataset_mixin import DiskIOMixin
from ...const import StrPath


class JSONDataset(DictDataset, DiskIOMixin):
    DEFAULT_CONFIG = {
        **DictDataset.DEFAULT_CONFIG,
        **DiskIOMixin.DEFAULT_CONFIG,
        'encoding': None,
    }

    def __init__(self, source: Dict[str, Any], fp=None, **kwargs):
        super().__init__(source, **kwargs)
        self.register_to_config(
            fp=fp,
            encoding=kwargs.get('encoding', None)
        )

    def commit(self, indent=4, **kwargs):
        return self.dump(self.config['fp'], indent=indent, **kwargs)

    @classmethod
    def from_disk(cls, fp, encoding='utf-8', **kwargs):
        if not os.path.exists(fp):
            data = {}
        else:
            with open(fp, 'r', encoding=encoding) as f:
                data = json.load(f)
        config = dict(fp=os.path.abspath(fp), encoding=encoding)
        kwargs = {**config, **kwargs}
        return cls(data, **kwargs)

    def dump(self, fp, mode='w', indent=4):
        if mode == 'a' and os.path.exists(fp):
            with open(fp, 'r') as f:
                data = json.load(f)
            data.update(self.data)
        else:
            data = self.data
        with open(fp, 'w') as f:
            json.dump(data, f, indent=indent)
