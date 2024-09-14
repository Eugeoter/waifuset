import math
import os
import gradio as gr
from typing import Union, Tuple, Dict, Iterable
from pathlib import Path
from waifuset import Dataset, ParasiteDataset, DictDataset
from waifuset import logging


class UISubset(ParasiteDataset):
    def __init__(self, source, host=None, page_size=None, **kwargs):
        assert page_size is not None, "page_size must be specified"
        super().__init__(source, host=host)
        self.page_size = page_size
        self.register_to_config(
            page_size=page_size,
        )

    def get_categories(self):
        if not hasattr(self, "categories"):
            self.categories = get_categories(self)
        return self.categories

    @property
    def num_pages(self):
        return int(math.ceil(len(self) / self.page_size))

    def page(self, i: int, **kwargs):
        r"""
        Get the i-th page of the dataset.
        """
        page_values = self[i * self.page_size: (i + 1) * self.page_size]
        return DictDataset({v['image_key']: v for v in page_values})


class UIDataset(UISubset):
    def __init__(self, source=None, host=None, page_size=None, **kwargs):
        super().__init__(source, host=host, page_size=page_size, **kwargs)
        self.curset = self.fullset

    @ property
    def fullset(self):
        return UISubset.from_dataset(self.root, host=self)

    @ property
    def header(self):
        return self.root.header

    @ property
    def types(self):
        return self.root.types

    @ property
    def info(self):
        return self.root.info

    def change_curset(self, subset=None):
        if isinstance(subset, UISubset):
            self.curset = subset
        elif isinstance(subset, Iterable):
            self.curset = UISubset.from_dataset(subset, self)
        else:
            raise ValueError(f"subset must be a list or a Dataset, not {type(subset)}")

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


def get_root(dataset: Dataset):
    return dataset.root if hasattr(dataset, 'root') else dataset


def get_categories(dataset: Dataset):
    rootset = get_root(dataset)

    # find image keys without category
    if 'category' in dataset.header:
        cats = set(cat for cat in dataset.kvalues('category', distinct=True) if cat is not None)
        # select images without category
        if rootset.__class__.__name__ == 'SQLite3Dataset':
            uncat_keys = [v[0] for v in rootset.table.select_is('category', None)]  # search keys without category in the root sqlite3 dataset
            uncat_keys = [key for key in uncat_keys if key in dataset]  # and intersect with the dataset
        else:
            uncat_keys = [k for k, v in dataset.items() if v.get('category') is None]  # search keys without category in the dataset
        uncat_dataset = ParasiteDataset.from_keys(uncat_keys, host=rootset)
    else:
        cats = set()
        uncat_dataset = dataset.keys()  # all images are uncat

    if uncat_keys:
        for img_key, img_path in uncat_dataset.kitems('image_path'):
            cat = Path(img_path).parent.name
            dataset.set(img_key, {'category': cat})
            if cat not in cats:
                cats.add(cat)
    cats = list(sorted(cats))
    return cats
