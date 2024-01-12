import gradio as gr
import os
import json
import pickle
import time
import math
import functools
from pathlib import Path
from tqdm import tqdm
from typing import List, Iterable, Dict, Any, Callable, Union, Tuple
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


class ChunkedDataset(Dataset):
    def __init__(self, source, *args, chunk_size=None, **kwargs):
        super().__init__(source, *args, **kwargs)
        self.chunk_size = chunk_size

    def chunk(self, index) -> Dataset:
        if self.chunk_size is None:
            return self
        elif index < 0 or index >= self.num_chunks:
            return Dataset()
        return Dataset(self.values()[index * self.chunk_size: (index + 1) * self.chunk_size])

    @property
    def num_chunks(self):
        return math.ceil(len(self) / self.chunk_size) if self.chunk_size is not None else 1


class TagTable:
    def __init__(self):
        self._table: Dict[str, set] = {}

    def query(self, tag):
        return self._table.get(tag, set())

    def remove_key(self, key):
        for tag, key_set in self._table.items():
            if key in key_set:
                key_set.remove(key)

    def add(self, tag, key):
        if tag not in self._table:
            self._table[tag] = set()
        self._table[tag].add(key)

    def remove(self, tag, key):
        if tag not in self._table:
            return
        self._table[tag].remove(key)
        if len(self._table[tag]) == 0:
            del self._table[tag]

    def __contains__(self, tag):
        return tag in self._table

    def __getitem__(self, tag):
        return self._table[tag]


class UIDataset(ChunkedDataset):
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

        # self.subsets = None
        self.categories = sorted(list(set(img_info.category for img_info in self.values()))) if len(self) > 0 else []
        self.tag_table = None
        self.selected = SelectData()
        self.history = History()
        self.buffer = Dataset()

        # self.init_subsets()

        if formalize_caption:
            self.buffer.update(self)

    # def init_subsets(self):
    #     subsets = {}
    #     for k, v in tqdm(self.items(), desc='making subsets'):
    #         if v.category not in subsets:
    #             subsets[v.category] = ChunkedDataset(source=None, chunk_size=self.chunk_size)
    #         subsets[v.category][k] = v
    #     self.subsets = subsets

    def init_tag_table(self, subset_key=None):
        if self.tag_table is not None:
            return
        self.tag_table = TagTable()
        for image_key, image_info in tqdm(self.items(), desc='initializing tag table'):
            if not image_info.caption:
                continue
            for tag in image_info.caption:
                self.tag_table.add(tag, image_key)

    def make_subset(self, condition: Callable[[ImageInfo], bool], chunk_size=None, *args, **kwargs):
        return ChunkedDataset(self, chunk_size=chunk_size, *args, condition=condition, **kwargs)

    def select(self, selected: Union[gr.SelectData, Tuple[int, str]]):
        if isinstance(selected, gr.SelectData):
            self.selected.index = selected.index
            image_key = os.path.basename(os.path.splitext(selected.value['image']['orig_name'])[0])
            self.selected.image_key = image_key
        elif isinstance(selected, tuple):
            self.selected.index, self.selected.image_key = selected
        else:
            raise NotImplementedError

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

    # core setitem method
    def __setitem__(self, key, value):
        img_info = self.get(key)

        # update tag table
        if img_info and self.tag_table:
            for tag in value.caption - img_info.caption:  # introduce new tags
                self.tag_table.add(tag, key)
            for tag in img_info.caption - value.caption:  # remove old tags
                self.tag_table.remove(tag, key)

        super().__setitem__(key, value)

        # # update subset
        # subset = self.subsets[value.category]
        # subset[key] = value

        # update buffer
        self.buffer[key] = value

    # core delitem method
    def pop(self, key, default=None):
        img_info = super().pop(key, default)

        # # update subset
        # subset = self.subsets[img_info.category]
        # del subset[key]

        # update tag table
        if self.tag_table is not None:
            self.tag_table.remove_key(key)

        # update buffer
        self.buffer[key] = img_info

        return img_info

    # setitem with updating history
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

    # delitem with updating history
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
            logu.info(f'Saving dataset...')

        if self.write_to_database:
            if self.verbose:
                tic = time.time()
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
            if self.verbose:
                toc = time.time()
                time_cost1 = toc - tic

        if self.write_to_txt:
            if self.verbose:
                tic = time.time()
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
                toc = time.time()
                time_cost2 = toc - tic

        if self.verbose:
            logu.info(f"revised total {len(self.buffer)} items.")
        self.buffer.clear()

        if self.verbose:
            toc = time.time()
            if self.write_to_database:
                logu.success(f'Write to database: saved to `{logu.yellow(self.database_file)}`: time_cost={time_cost1:.2f}s.')
            if self.write_to_txt:
                logu.success(f'Write to txt: time_cost={time_cost2:.2f}s.')
