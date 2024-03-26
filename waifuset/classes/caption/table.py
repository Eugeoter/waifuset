import json
from typing import Dict, List, Union
from tqdm import tqdm
from pathlib import Path
from .caption import preprocess_tag


class Tag:
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        return self.tag

    def __repr__(self):
        return self.tag

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.tag)


class Character(Tag):
    def __init__(self, tag, *args, **kwargs):
        super().__init__(tag, *args, **kwargs)
        self.counter: Dict[Tag, int] = {}

    def update(self, tags: List[Union[str, Tag]]):
        for tag in tags:
            if tag.startswith('character:'):
                tag = tag.split(':')[1].strip()
            tag = preprocess_tag(tag)
            if isinstance(tag, str):
                tag = Tag(tag)
            self.counter[tag] = self.counter.get(tag, 0) + 1


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
                char_tag = preprocess_tag(char_tag)
                char = count_table.get(char_tag)
                if char is None:
                    char = Character(char_tag)
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
        freqs = {tag: freq for tag, freq in freqs.items() if any(regex.match(preprocess_tag(tag)) for regex in tagging.REGEX_CHARACTER_FEATURES)}
        feature_table[char_tag] = set(freqs.keys())
    return feature_table


def freq_table_to_feature_table(freq_table, freq_thres=0.3):
    from . import tagging
    feature_table = {}
    for char_tag, counter in tqdm(freq_table.items(), desc='make feature table', disable=True):
        freqs = {tag: freq for tag, freq in counter.items() if freq >= freq_thres}
        freqs = {tag: freq for tag, freq in freqs.items() if any(regex.match(preprocess_tag(tag)) for regex in tagging.REGEX_CHARACTER_FEATURES)}
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
        table = {preprocess_tag(char_tag): set(preprocess_tag(tag) for tag in tags) for char_tag, tags in table.items() if tags}
        self.table = table
        self.freq_thres = freq_thres
        self.count_thres = count_thres
        self.least_sample_size = least_sample_size

    def __getitem__(self, character):
        char_tag = preprocess_tag(character)
        return self.table[char_tag]

    def get(self, character, default=None):
        char_tag = preprocess_tag(character)
        return self.table.get(char_tag, default)

    def items(self):
        return self.table.items()

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()
