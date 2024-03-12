import gradio as gr
import re
import math
import os
import json
import time
from tqdm import tqdm
from typing import Union, Callable, Tuple
from pathlib import Path
from typing import List, Dict
from ..classes import Dataset, ImageInfo, Caption
from ..classes.caption.caption import preprocess_tag
from ..utils import log_utils as logu
from ..utils.file_utils import listdir


class UISelectData:
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


class UIEditHistory:
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

        # DEBUG
        # print(f"history[{img_key}]:")
        # print(f"  [Z]:")
        # for i, img_info in enumerate(self._z[img_key]):
        #     info_dict = img_info.dict() if img_info is not None else None
        #     info_dict = '\n'.join([f'    {k}: {v}' for k, v in info_dict.items()]) if info_dict is not None else None
        #     print(f"    [{i}]: {info_dict}")

        # print(f"  [Y]:")
        # for i, img_info in enumerate(self._y[img_key]):
        #     info_dict = img_info.dict() if img_info is not None else None
        #     info_dict = '\n'.join([f'    {k}: {v}' for k, v in info_dict.items()]) if info_dict is not None else None
        #     print(f"    [{i}]: {info_dict}")

    def undo(self, img_key):
        if img_key not in self._z or len(self._z[img_key]) <= 1:
            return LAMBDA
        self._y[img_key].append(self._z[img_key].pop())
        img_info = self._z[img_key][-1]

        # DEBUG
        # print(f"undo return:")
        # info_dict = img_info.dict() if img_info is not None else None
        # info_dict = '\n'.join([f'    {k}: {v}' for k, v in info_dict.items()]) if info_dict is not None else None
        # print(f"  {info_dict}")

        return img_info

    def redo(self, img_key):
        if img_key not in self._y or len(self._y[img_key]) <= 0:
            return LAMBDA
        img_info = self._y[img_key].pop()
        self._z[img_key].append(img_info)

        # DEBUG
        # print(f"redo return:")
        # info_dict = img_info.dict() if img_info is not None else None
        # info_dict = '\n'.join([f'    {k}: {v}' for k, v in info_dict.items()]) if info_dict is not None else None
        # print(f"  {info_dict}")

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


class UISampleHistory:
    def __init__(self):
        self._lst = []
        self._set = set()
        self.index = None

    def add(self, image_key):
        self._lst.append(image_key)
        self._set.add(image_key)

    def select(self, index, correct_index=True):
        if len(self._lst) == 0:
            return None
        if correct_index:
            index = min(max(index, 0), len(self._lst) - 1)
        elif index < 0 or index >= len(self._lst):
            return None
        self.index = index
        return self._lst[index]

    def __contains__(self, image_key):
        return image_key in self._set

    def __len__(self):
        return len(self._lst)


class TagPriority:
    def __init__(self, name, patterns: List[str], priority: int):
        assert isinstance(name, str), f"name must be str, but got {type(name)}."
        assert isinstance(patterns, list), f"patterns must be list, but got {type(patterns)}."
        assert isinstance(priority, int), f"priority must be int, but got {type(priority)}."
        self._name = name
        self._patterns = patterns
        self._priority = priority

        # cache
        self._pattern = None
        self._regex = None

    def clean_cache(self):
        self._pattern = None
        self._regex = None

    @property
    def patterns(self) -> List[str]:
        return self._patterns

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def pattern(self) -> str:
        if self._pattern is None:
            self._pattern = '|'.join(self._patterns).replace(' ', r'[\s_]')
        return self._pattern

    @property
    def regex(self) -> re.Pattern:
        if self._regex is None:
            self._regex = re.compile(self.pattern)
        return self._regex


class UITagPriorityManager:
    def __init__(self, priority: Dict[str, TagPriority] = {}):
        self._priority = {name: tp if isinstance(tp, TagPriority) else TagPriority(name, tp, i) for i, (name, tp) in enumerate(priority.items())}
        self._priority_regex = None

    @property
    def priority_regex(self) -> List[re.Pattern]:
        if self._priority_regex is None:
            sorted_tps = sorted([tp for tp in self._priority.values()], key=lambda tp: tp.priority)
            self._priority_regex = [tp.regex for tp in sorted_tps]
        return self._priority_regex

    @property
    def config(self):
        from .. import tagging
        return {name: [pattern for pattern in tp.patterns if pattern not in tagging.PRIORITY[name]] for name, tp in self._priority.items()}

    def __getitem__(self, name):
        return self._priority[name]

    def __setitem__(self, name, value):
        self._priority[name] = value
        self._priority_regex = None

    def __contains__(self, name):
        return name in self._priority

    def __len__(self):
        return len(self._priority)

    def keys(self):
        return self._priority.keys()

    def values(self):
        return self._priority.values()

    def items(self):
        return self._priority.items()


class UIChunkedDataset(Dataset):
    chunk_size: int

    def __init__(self, source=None, *args, chunk_size=None, **kwargs):
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


class UITab:
    def __init__(self, tab: gr.Tab):
        self._tab = tab

    @property
    def tab(self):
        return self._tab

    @tab.setter
    def tab(self, value):
        self._tab = value


class UITagTable:
    def __init__(self):
        self._table: Dict[str, set] = {}

    def query(self, tag):
        return self._table.get(tag, set())

    def remove_key(self, key):
        for tag, key_set in self._table.items():
            if key in key_set:
                key_set.remove(key)

    def add(self, tag, key, preprocess=True):
        if preprocess:
            tag = preprocess_tag(tag)
        if tag not in self._table:
            self._table[tag] = set()
        self._table[tag].add(key)
        # if 'jinx' in tag:
        #     print(f"table[{tag}] += {key}")

    def remove(self, tag, key, preprocess=True):
        if preprocess:
            tag = preprocess_tag(tag)
        if tag not in self._table:
            return
        self._table[tag].remove(key)
        if len(self._table[tag]) == 0:
            del self._table[tag]
        # if 'jinx' in tag:
        #     print(f"table[{tag}] -= {key}")

    def __contains__(self, tag):
        return tag in self._table

    def __getitem__(self, tag):
        return self._table[tag]

    def __len__(self):
        return len(self._table)

    def keys(self):
        return self._table.keys()

    def items(self):
        return self._table.items()

    def values(self):
        return self._table.values()


class UIDataset(UIChunkedDataset):
    selected: UISelectData
    edit_history: UIEditHistory

    def __init__(self, source=None, write_to_database=False, write_to_txt=False, database_file=None, backup_dir=None, *args, **kwargs):
        if write_to_database and database_file is None:
            raise ValueError("database file must be specified when write_to_database is True.")

        self.write_to_database = write_to_database
        self.write_to_txt = write_to_txt
        self.database_file = Path(database_file).absolute() if database_file else None
        self.backup_dir = Path(backup_dir or './backups').absolute()

        same_io = source and len(source) == 1 and write_to_database and self.database_file.samefile(source[0])
        if same_io:
            print(f"[ui_dataset] same io")
            super().__init__(source, *args, **kwargs)
        else:
            print(f"[ui_dataset] overload: {logu.yellow(source)} -> {logu.yellow(self.database_file)}")
            if self.database_file and self.database_file.is_file():
                database = Dataset(self.database_file, *args, **kwargs)
            else:
                database = None

            super().__init__(source, *args, cacheset=database, **kwargs)

        # self.subsets = None
        self.buffer = Dataset()
        self.categories = sorted(list(set(img_info.category for img_info in self.values()))) if len(self) > 0 else []
        self.tag_table = None
        self.selected = UISelectData()
        self.edit_history = UIEditHistory()
        self.subset = self

        if kwargs.get('formalize_caption', False):
            self.buffer.update(self)
        elif not same_io or write_to_txt:
            for img_key in tqdm(self, desc='initializing buffer'):
                if img_key not in database:
                    self.buffer[img_key] = self[img_key]

    def make_subset(self, *args, **kwargs):
        kwargs['cls'] = UIChunkedDataset
        return super().make_subset(*args, **kwargs)

    def init_tag_table(self):
        if self.tag_table is not None:
            return
        self.tag_table = UITagTable()
        for image_key, image_info in tqdm(self.items(), desc='initializing tag table'):
            caption = image_info.caption
            if caption is None:
                continue
            for tag in caption:
                self.tag_table.add(tag, image_key)

    def select(self, selected: Union[gr.SelectData, Tuple[int, str]]):
        if isinstance(selected, gr.SelectData):
            self.selected.index = selected.index
            image_filename = selected.value['image']['orig_name']
            image_key = os.path.basename(os.path.splitext(image_filename)[0])
            self.selected.image_key = image_key
        elif isinstance(selected, tuple):
            self.selected.index, self.selected.image_key = selected
        elif selected is None:
            self.selected.index, self.selected.image_key = None, None
        else:
            raise NotImplementedError

        # print(f"selected: {self.selected.index}, {self.selected.image_key}")

        return self.selected.image_key

    def undo(self, image_key):
        image_info = self.edit_history.undo(image_key)
        if image_info is LAMBDA:  # empty history, return original
            return self[image_key]
        elif image_info is not None:  # undo
            self[image_key] = image_info
        else:  # is deletion operation, then delete
            del self[image_key]
        return image_info

    def redo(self, image_key):
        image_info = self.edit_history.redo(image_key)
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
        # print(f"setitem {key} from {img_info.caption} to {value.caption}")

        # update tag table
        if self.tag_table is not None:
            orig_caption = Caption() if img_info is None or img_info.caption is None else img_info.caption.copy()
            new_caption = Caption() if value is None or value.caption is None else value.caption.copy()
            orig_caption.tags = [preprocess_tag(tag) for tag in orig_caption.tags]
            new_caption.tags = [preprocess_tag(tag) for tag in new_caption.tags]
            # print(f"add tags: {new_caption - orig_caption}")
            for tag in new_caption - orig_caption:  # introduce new tags
                self.tag_table.add(tag, key, preprocess=False)
            # print(f"remove tags: {orig_caption - new_caption}")
            for tag in orig_caption - new_caption:  # remove old tags
                self.tag_table.remove(tag, key, preprocess=False)

        super().__setitem__(key, value)

        # update subset
        if not self.subset is self:
            self.subset[key] = value

        # update buffer
        self.buffer[key] = value

    # core delitem method
    def pop(self, key, default=None):
        img_info = super().pop(key, default)

        # update subset
        if not self.subset is self:
            del self.subset[key]

        # update tag table
        if self.tag_table is not None:
            self.tag_table.remove_key(key)

        # update buffer
        self.buffer[key] = img_info

        return img_info

    # setitem with updating history
    def set(self, key, value):
        # print(f"set {key}")
        if self[key] == value:
            return

        # init history if needed
        if key not in self.edit_history:
            self.edit_history.init(key, self.get(key).copy())

        # update dataset
        self[key] = value

        # record history
        self.edit_history.record(key, value.copy())

    # delitem with updating history
    def remove(self, key):
        # pop from dataset
        img_info = self.pop(key)

        # record
        if key not in self.edit_history:
            self.edit_history.init(key, img_info)
        self.edit_history.record(key, None)

        return img_info

    def save(self, progress=gr.Progress(track_tqdm=True)):
        if self.verbose:
            logu.info(f'saving dataset: {len(self.buffer)}/{len(self)}')

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
                    img_info.write_txt_caption()
                else:
                    backup(img_info)
            if self.verbose:
                toc = time.time()
                time_cost2 = toc - tic

        self.buffer.clear()

        if self.verbose:
            toc = time.time()
            if self.write_to_database:
                logu.success(f'write to database: `{logu.yellow(self.database_file)}` | time_cost={time_cost1:.2f}s.')
            if self.write_to_txt:
                logu.success(f'write to txt | time_cost={time_cost2:.2f}s.')


def backup(image_info):
    img_path = image_info.image_path
    cap_path = str(img_path.with_suffix('.txt'))
    img_path = str(img_path)

    img_bkup_path = img_path + '.bak'
    cap_bkup_path = cap_path + '.bak'

    if os.path.isfile(img_path):
        if os.path.isfile(img_bkup_path):
            os.remove(img_bkup_path)
        os.rename(img_path, img_bkup_path)

    if os.path.isfile(cap_path):
        if os.path.isfile(cap_bkup_path):
            os.remove(cap_bkup_path)
        os.rename(cap_path, cap_bkup_path)
