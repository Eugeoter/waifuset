import json
from typing import Dict, List, Union
from tqdm import tqdm
from pathlib import Path
from .caption import fmt2standard


class StandardTag:
    r"""
    Base class for standard tags.
    """

    def __init__(self, tag):
        self.tag = fmt2standard(tag)
        self.counter: Dict[StandardTag, int] = {}

    def __str__(self):
        return self.tag

    def __repr__(self):
        return self.tag

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.tag)

    def update(self, tags: List[Union[str, 'StandardTag']]):
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if isinstance(tag, str):
                tag = self.__class__(tag)
            self.counter[tag] = self.counter.get(tag, 0) + 1


class SingleInstanceStandardTag(StandardTag):
    _instance = {}

    def __new__(cls, tag):
        tag = fmt2standard(tag)
        if tag not in cls._instance:
            cls._instance[tag] = super().__new__(cls)
            print(f"New instance: {tag}")
        return cls._instance[tag]


class StandardTable:
    r"""
    Base class for standard tables which automatically convert dict keys to standard format.
    """

    def __init__(self) -> None:
        self.table = {}

    def __getitem__(self, key):
        return self.table[fmt2standard(key)]

    def __setitem__(self, key, value):
        self.table[fmt2standard(key)] = value

    def __delitem__(self, key):
        del self.table[fmt2standard(key)]

    def __contains__(self, key):
        return fmt2standard(key) in self.table

    def get(self, key, default=None):
        return self.table.get(fmt2standard(key), default)

    def items(self):
        return self.table.items()

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()

    def update(self, other):
        for key, value in other.items():
            self[key] = value


def dataset_to_count_table(dataset):
    count_table = {}
    for img_key, img_info in tqdm(dataset.items(), desc='make count table', disable=False):
        caption = img_info['caption']
        if not caption:
            continue
        tags = caption.split(', ') if isinstance(caption, str) else caption.tags
        for tag in tags:
            if tag.startswith('character:'):
                char_tag = tag.split(':')[1].strip()
                char_tag = fmt2standard(char_tag)
                char = count_table.get(char_tag)
                if char is None:
                    char = StandardTag(char_tag)
                    count_table[char_tag] = char
                char.update(tags)
    # sort
    for char_tag, char in count_table.items():
        counter = {tag: count for tag, count in sorted(char.counter.items(), key=lambda x: x[1], reverse=True)}
        count_table[char_tag] = counter
    # jsonize
    count_table = {char_tag: {str(tag): count for tag, count in counter.items()} for char_tag, counter in count_table.items()}
    return count_table


def count_table_to_feature_table(count_table, freq_thres=0.3, count_thres=1, least_sample_size=50):
    from . import tagging
    feature_table = {}
    for char_tag, counter in tqdm(count_table.items(), desc='make feature table', disable=True):
        total = counter[char_tag]
        if total < least_sample_size:
            continue
        freqs = {tag: count / total for tag, count in counter.items() if count >= count_thres}
        freqs = {tag: freq for tag, freq in freqs.items() if freq >= freq_thres}
        freqs = {tag: freq for tag, freq in freqs.items() if any(regex.match(fmt2standard(tag)) for regex in tagging.REGEX_CHARACTER_FEATURES)}
        feature_table[char_tag] = set(freqs.keys())
    return feature_table


def freq_table_to_feature_table(freq_table, freq_thres=0.3):
    from . import tagging
    feature_table = {}
    for char_tag, counter in tqdm(freq_table.items(), desc='make feature table', disable=True):
        freqs = {tag: freq for tag, freq in counter.items() if freq >= freq_thres}
        freqs = {tag: freq for tag, freq in freqs.items() if any(regex.match(fmt2standard(tag)) for regex in tagging.REGEX_CHARACTER_FEATURES)}
        feature_table[char_tag] = set(freqs.keys())
    return feature_table


def get_table_type(table):
    v0 = next(iter(table.values()))
    if isinstance(v0, (list, set, tuple)):
        return 'feature_table'
    elif isinstance(v0, dict):
        v0v0 = next(iter(v0.values()))
        if isinstance(v0v0, int):
            return 'count_table'
        elif isinstance(v0v0, float):
            return 'freq_table'
    return None


class FeatureTable:
    r"""
    A table records the core features of characters.
    """

    def __init__(self, source, freq_thres=0.3, count_thres=1, least_sample_size=50):
        if isinstance(source, (str, Path)) and Path(source).suffix == '.json':
            with open(source, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
            if (table_type := get_table_type(dataset)):
                if table_type == 'count_table':
                    table = count_table_to_feature_table(dataset, freq_thres=freq_thres, count_thres=count_thres, least_sample_size=least_sample_size)
                elif table_type == 'freq_table':
                    table = freq_table_to_feature_table(dataset, freq_thres=freq_thres)
                elif table_type == 'feature_table':
                    table = dataset
            else:
                table = dataset_to_count_table(dataset)
                table = count_table_to_feature_table(table, freq_thres=freq_thres, count_thres=count_thres, least_sample_size=least_sample_size)
        else:
            from ..dataset.dataset import Dataset
            dataset = Dataset(source)
            table = dataset_to_count_table(dataset)
            table = count_table_to_feature_table(dataset, freq_thres=freq_thres, count_thres=count_thres, least_sample_size=least_sample_size)
        # formalize table
        table = {fmt2standard(char_tag): set(fmt2standard(tag) for tag in tags) for char_tag, tags in table.items() if tags}
        self.table = table
        self.freq_thres = freq_thres
        self.count_thres = count_thres
        self.least_sample_size = least_sample_size

    def __getitem__(self, character):
        char_tag = fmt2standard(character)
        return self.table[char_tag]

    def get(self, character, default=None):
        char_tag = fmt2standard(character)
        return self.table.get(char_tag, default)

    def items(self):
        return self.table.items()

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()
