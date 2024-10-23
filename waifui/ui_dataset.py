import math
import os
import gradio as gr
from typing import Union, Tuple, Dict, Iterable, List
from pathlib import Path
from waifuset import Dataset, ParasiteDataset, DictDataset
from waifuset import logging

logger = logging.get_logger('UI')


class UISubset(ParasiteDataset):
    page_size: int
    categories: List[str]

    def __init__(self, source, host=None, page_size=None, is_full=False, **kwargs):
        # check inputs
        assert page_size is not None, "page_size must be specified"
        assert isinstance(page_size, int), "page_size must be an integer"
        assert page_size > 0, "page_size must be greater than 0"

        super().__init__(source, host=host, **kwargs)
        self.page_size = page_size
        self.is_full = is_full
        self.categories = None
        self.register_to_config(
            page_size=page_size,
        )

    def __getitem__(self, key):
        if self.is_full:
            return self.root[key]
        else:
            return super().__getitem__(key)

    def get_categories(self) -> List[str]:
        if self.categories is None:
            self.categories = get_categories(self)
        return self.categories

    def set_categories(self, categories: Iterable[str]):
        self.categories = list(categories)

    @property
    def num_pages(self) -> int:
        return int(math.ceil(len(self) / self.page_size))

    def get_page(self, i: int, **kwargs) -> DictDataset:
        r"""
        Get the i-th page of the dataset.
        """
        page_values = self[i * self.page_size: (i + 1) * self.page_size]
        return DictDataset({v['image_key']: v for v in page_values})


class UIDataset(UISubset):
    r"""
    UIDataset controls the current subset of the dataset to be displayed. The current dataset have to be a UISubset object.
    """

    def __init__(self, source=None, host=None, page_size=None, is_full=False, **kwargs):
        super().__init__(source, host=host, page_size=page_size, is_full=is_full, **kwargs)
        self.fullset: UISubset = None
        self.curset: UISubset = None

    def get_curset(self):
        if self.curset is None:
            self.curset = self.get_fullset()
        return self.curset

    def get_fullset(self):
        if self.fullset is None:
            self.fullset = UISubset.from_dataset(self.root, host=self, is_full=True)
        return self.fullset

    @property
    def headers(self):
        return self.root.headers

    @property
    def column_names(self):
        return self.root.column_names

    @property
    def types(self):
        return self.root.types

    @property
    def info(self):
        return self.root.info

    def set_curset(self, subset: UISubset = None):
        r"""
        Set the current dataset to a new UISubset.
        """
        if isinstance(subset, UISubset):
            self.curset = subset
        else:
            raise ValueError(f"subset must be an instance of UISubset, but got {type(subset)}")

    def select(self, selected: Union[gr.SelectData, Tuple[int, str]]):
        r"""
        Set the current selected data to a new one.

        @param selected: The new selected data. It can be one of the following types:
            - gr.SelectData: The selected data of the gr.Gallery object.
            - Tuple[int, str]: The [index, image_key] pair of the selected data.
            - None: Empty the selected data.
        """
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
            raise ValueError(f"selected must be an instance of gr.SelectData or Tuple[int, str], but got {type(selected)}")


def get_root(dataset: Dataset):
    return dataset.root if hasattr(dataset, 'root') else dataset


def get_categories(dataset: Dataset):
    rootset = get_root(dataset)

    logger.info(f"Getting all the categories from {logging.yellow(len(dataset))} data...")

    # find image keys without category
    if 'category' in dataset.headers:
        cats = set(cat for cat in dataset.kvalues('category', distinct=True) if cat)  # get all categories
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
