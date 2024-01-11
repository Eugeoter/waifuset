import gradio as gr
import os
import json
import pickle
import time
import functools
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


LAMBDA = -1


class History:
    def __init__(self):
        self._z = {}
        self._y = {}

    def init(self, img_key, img_info):
        if img_key not in self._z:
            self._z[img_key] = [img_info.copy() if img_info else None]
            self._y[img_key] = []

    def record(self, img_key, img_info):
        if img_key not in self._z:
            raise ValueError(f"img_key={img_key} not in history. You should init by `init(img_key, img_info)` first.")
        self._z[img_key].append(img_info.copy() if img_info else None)
        self._y[img_key] = []
        # print(f"history[{img_key}]: \n{[f'{i.key}: {i.caption}' if i is not None else None for i in self._z[img_key]]}\n{[f'{i.key}: {i.caption}' if i is not None else None for i in self._y[img_key]]}")

    def undo(self, img_key):
        if img_key not in self._z or len(self._z[img_key]) <= 1:
            return LAMBDA
        self._y[img_key].append(self._z[img_key].pop())
        img_info = self._z[img_key][-1]
        # print(f"undo return: {f'{img_info.key}: {img_info.caption}' if img_info is not None else None}")
        return img_info

    def redo(self, img_key):
        if img_key not in self._y or len(self._y[img_key]) <= 0:
            return LAMBDA
        img_info = self._y[img_key].pop()
        self._z[img_key].append(img_info)
        # print(f"redo return: {f'{img_info.key}: {img_info.caption}' if img_info is not None else None}")
        return img_info

    def items(self):
        return {img_key: img_infos[-1] for img_key, img_infos in self._z.items() if img_infos[-1] is not None}.items()

    def keys(self):
        return self._z.keys()

    def values(self):
        return [img_infos[-1] for img_infos in self._z.values() if img_infos[-1] is not None]

    def __contains__(self, key):
        return key in self._z

    def __len__(self):
        return len(self._z)


class UIDataset(Dataset):
    selected: SelectData
    history: History

    def __init__(self, source=None, write_to_database=False, write_to_txt=False, database_file=None, formalize_caption=False, *args, **kwargs):
        if write_to_database and database_file is None:
            raise ValueError("database file must be specified when write_to_database is True.")

        self.write_to_database = write_to_database
        self.write_to_txt = write_to_txt
        self.database_file = Path(database_file).absolute() if database_file else None

        if isinstance(source, (str, Path)) and source.endswith(".pkl"):
            with open(source, 'rb') as f:
                self._data = pickle.load(f)._data
        else:
            super().__init__(source, *args, **kwargs)

        self.selected = SelectData()
        self.history = History()
        self.buffer = Dataset()

        self.make_subsets()

        if formalize_caption:
            self.buffer.update(self)

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

    def undo(self, image_key):
        image_info = self.history.undo(image_key)
        if image_info is LAMBDA:  # empty history, return original
            return self[image_key]
        elif image_info is not None:  # undo
            self[image_key] = image_info
        else:  # is deletion operation, then delete
            del self[image_key]
        return image_info

    def redo(self, image_key):
        image_info = self.history.redo(image_key)
        if image_info is LAMBDA:  # empty history, return original
            return self[image_key]
        elif image_info:  # redo
            self[image_key] = image_info
        else:  # is deletion operation, then delete
            del self[image_key]
        return image_info

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        # update subset
        subset = self.subsets[value.category]
        subset[key] = value

        # update buffer
        self.buffer[key] = value

    def pop(self, key, default=None):
        img_info = super().pop(key, default)

        # update subset
        subset = self.subsets[img_info.category]
        del subset[key]

        # update buffer
        self.buffer[key] = None

        return img_info

    def set(self, key, value):
        if self[key] == value:
            return

        # init history if needed
        if key not in self.history:
            self.history.init(key, self.get(key))

        # update dataset
        self[key] = value

        # record
        self.history.record(key, value)

    def remove(self, key):
        # pop from dataset
        img_info = self.pop(key)

        # record
        if key not in self.history:
            self.history.init(key, img_info)
        self.history.record(key, None)

        return img_info

    def save(self, progress=gr.Progress(track_tqdm=True)):
        if self.verbose:
            tic = time.time()
            logu.info(f'Saving dataset...')

        if self.write_to_database:
            if not self.database_file.is_file():  # dump all
                self.database_file.parent.mkdir(parents=True, exist_ok=True)
                self.to_json(self.database_file)
            else:  # dump history only
                with open(self.database_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                for img_key, img_info in tqdm(self.buffer.items(), desc='dumping to database', disable=not self.verbose):
                    if img_key in self:
                        json_data[img_key] = img_info.dict()
                    elif img_key in json_data:
                        del json_data[img_key]
                with open(self.database_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False, sort_keys=False)

        if self.write_to_txt:
            for img_key, img_info in tqdm(self.buffer.items(), desc='dumping to txt', disable=not self.verbose):
                img_info: ImageInfo
                if img_key in self:
                    img_info.write_caption()
                else:
                    img_path = img_info.image_path
                    img_path.unlink()
                    cap_path = img_info.with_suffix('.txt')
                    if cap_path.is_file():
                        cap_path.unlink()

        if self.verbose:
            logu.info(f"revised total {len(self.buffer)} items.")
        self.buffer.clear()

        if self.verbose:
            toc = time.time()
            logu.success(f'Dataset saved to `{logu.yellow(self.database_file)}`: time_cost={toc - tic:.2f}s.')
