import gradio as gr
import os
import json
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Iterable, Dict, Any
from ..classes import Dataset, ImageInfo
from ..utils import log_utils as logu


class SelectData:
    def __init__(self, index=None, image_key=None):
        self._index = index
        self._image_key = image_key

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def image_key(self):
        return self._image_key

    @image_key.setter
    def image_key(self, value):
        self._image_key = value


class UIDataset(Dataset):
    selected: SelectData
    history: Dict[str, ImageInfo]

    def __init__(self, source=None, write_to_database=False, write_to_txt=False, database_path=None, *args, **kwargs):
        if write_to_database and database_path is None:
            raise ValueError("database_path must be specified when write_to_database is True.")

        self.write_to_database = write_to_database
        self.write_to_txt = write_to_txt
        self.database_path = Path(database_path).absolute() if database_path else None

        if isinstance(source, (str, Path)) and source.endswith(".pkl"):
            with open(source, 'rb') as f:
                self._data = pickle.load(f)._data
        else:
            super().__init__(source, *args, **kwargs)

        self.selected = SelectData()
        self.history = {}

        self.make_subsets()

    def make_subsets(self):
        subsets = {}
        for k, v in tqdm(self.items(), desc='making subsets'):
            if v.category not in subsets:
                subsets[v.category] = Dataset()
            subsets[v.category][k] = v
        self.subsets = subsets

    def select(self, selected: gr.SelectData):
        if isinstance(selected, gr.SelectData):
            self.selected.index = selected.index
            image_key = os.path.basename(os.path.splitext(selected.value['image']['orig_name'])[0])
            self.selected.image_key = image_key

        elif isinstance(selected, str):  # selected is image_key
            self.selected.image_key = selected
            img_info = self[selected]
            subset = self.subsets[img_info.category]
            self.selected.index = list(subset.keys()).index(selected)

        elif isinstance(selected, int):  # selected is index
            pre_img_key = self.selected.image_key
            category = self[pre_img_key].category
            self.selected.index = selected
            self.selected.image_key = list(self.subsets[category].keys())[selected]

        elif selected is None:  # selected is None
            self.selected.index = None
            self.selected.image_key = None

        return self.selected.image_key

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.history[key] = value

    def __delitem__(self, key):
        self.history[key] = self[key]
        super().__delitem__(key)

    def save(self, progress=gr.Progress(track_tqdm=True)):
        if self.verbose:
            tic = time.time()
            logu.info(f'Saving dataset...')

        if self.write_to_database:
            if not self.database_path.is_file():  # dump all
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                json_data = {k: v.dict() for k, v in tqdm(self.items(), desc='dumping to database', disable=not self.verbose)}
                with open(self.database_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False, sort_keys=False)
            else:  # dump history only
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                for k, v in tqdm(self.history.items(), desc='dumping to database', disable=not self.verbose):
                    if k in self:
                        json_data[k] = v.dict()
                    elif k in json_data:
                        del json_data[k]
                with open(self.database_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False, sort_keys=False)

        if self.write_to_txt:
            for k, v in tqdm(self.history.items(), desc='dumping to txt', disable=not self.verbose):
                if k in self:
                    v.write_caption()
                else:
                    image_path = v.image_path
                    image_path.unlink()

        if self.verbose:
            logu.info(f"revised total {len(self.history)} items.")
        self.history.clear()

        if self.verbose:
            toc = time.time()
            logu.success(f'Dataset saved: time_cost={toc - tic:.2f}s.')
