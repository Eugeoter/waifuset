import csv
import os
from typing import Dict
from .dict_dataset import DictDataset
from .dataset_mixin import DiskIOMixin


class CSVDataset(DictDataset, DiskIOMixin):
    DEFAULT_CONFIG = {
        **DictDataset.DEFAULT_CONFIG,
        **DiskIOMixin.DEFAULT_CONFIG,
        'primary_key': None,
        'encoding': 'utf-8',
    }

    def __init__(self, source: Dict[str, Dict[str, str]], **kwargs):
        super().__init__(source, **kwargs)
        self.register_to_config(
            fp=kwargs.get('fp', None),
            primary_key=kwargs.get('primary_key', None),
            encoding=kwargs.get('encoding', None),
        )

    def commit(self, **kwargs):
        return self.dump(self.config['fp'], **kwargs)

    @classmethod
    def from_disk(cls, fp, primary_key=None, encoding='utf-8', **kwargs):
        if not os.path.exists(fp):
            data = {}
        else:
            with open(fp, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                primary_key = primary_key or header[0]
                data = {row[primary_key]: row for row in reader}
        config = dict(fp=os.path.abspath(fp), primary_key=primary_key, encoding=encoding)
        kwargs = {**config, **kwargs}
        return cls(data, **kwargs)

    def dump(self, fp, mode='w', encoding='utf-8'):
        if mode == 'a' and os.path.exists(fp):
            with open(fp, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                data = {row[self.fp_key]: row for row in reader}
            data.update(self.data)
        else:
            data = self.data
        with open(fp, 'w', encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(data.values())
