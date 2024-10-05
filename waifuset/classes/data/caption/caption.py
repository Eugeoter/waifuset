import re
from typing import Callable, Literal, List, Dict, Union, overload
from ..data import Data
from .... import tagging


class Caption(Data):
    def __new__(cls, text=None, sep=', ', **kwargs):
        if isinstance(text, Caption):
            text.sep = sep
            return text
        return super().__new__(cls)

    def __init__(self, text=None, sep=', ', **kwargs):
        self.sep = sep
        if isinstance(text, Caption):
            self = text
            return
        if isinstance(text, list):
            tags = text
        elif isinstance(text, str):
            tags = text.split(self.sep)
        elif text is None:
            tags = []
        else:
            raise ValueError(f"Invalid type for text: {type(text)}")
        self.tags = tags
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def attrs(self):
        d = {'tags': self.tags}
        for k, v in self.__dict__.items():
            if k.startswith('cache_'):
                d[k[6:]] = v
        return d

    @property
    def text(self):
        return self.sep.join(self.tags)

    def deduplicate(self):
        self.tags = deduplicate(self.tags)
        self._empty_cache()

    def deduplicated(self):
        caption = self.copy()
        caption.deduplicate()
        return caption

    @overload
    def format(self, fmt: Callable[[str], str]): ...

    @overload
    def format(self, fmt: Literal['danbooru']): ...

    def format(self, fmt):
        if isinstance(fmt, str):
            fmt = {
                'danbooru': tagging.fmt2danbooru,
            }[fmt]
        self.tags = [fmt(tag) for tag in self.tags]
        self._empty_cache()

    def formatted(self, fmt):
        caption = self.copy()
        caption.format(fmt)
        return caption

    def parse(self):
        r"""
        According to the danbooru wiki, extract artist, characters, and styles tags.
        """
        d = {}
        for i, tag in list(enumerate(self.tags)):
            if tagtype := tagging.get_tagtype_from_wiki(tag):
                self.tags[i] = tagging.comment_tag(tag, tagtype=tagtype)
                d.setdefault(tagtype, []).append(tag)
            elif tagtype := tagging.get_tagtype_from_tag(tag):
                d.setdefault(tagtype, []).append(tag)
        for tagtype in tagging.ALL_TAG_TYPES:
            setattr(self, tagtype, d.get(tagtype, []))

    def parsed(self):
        caption = self.copy()
        caption.parse()
        return caption

    @property
    def metadata(self):
        return {tagtype: getattr(self, tagtype) for tagtype in tagging.ALL_TAG_TYPES}

    def defeature(self, feature_type_to_frequency_threshold: Dict[Literal['physics', 'clothes', 'sex'], float] = tagging.DEFAULT_FEATURE_TYPE_TO_FREQUENCY_THRESHOLD):
        r"""
        According to the feature table which is extracted from danbooru wiki, remove feature tags of every characters.
        """
        if not self.character:
            return
        all_features = set()
        for character in self.character:
            all_features |= set(tagging.get_character_features(character, feature_type_to_frequency_threshold=feature_type_to_frequency_threshold))
        self.tags = [tag for tag in self.tags if tagging.fmt2danbooru(tag) not in all_features]  # defeature won't change properties

    def defeatured(self, feature_type_to_frequency_threshold: Dict[Literal['physics', 'clothes', 'sex'], float] = tagging.DEFAULT_FEATURE_TYPE_TO_FREQUENCY_THRESHOLD):
        caption = self.copy()
        caption.defeature(feature_type_to_frequency_threshold=feature_type_to_frequency_threshold)
        return caption

    def sort(self, key=None, reverse=False):
        r"""
        Sort tags by priority. If key is `None`, use default priority.
        """
        key = key or tagging.get_tag_priority
        self.tags.sort(key=key, reverse=reverse)

    def sorted(self, key=None, reverse=False):
        caption = self.copy()
        caption.sort(key=key, reverse=reverse)
        return caption

    def deimplicate(self):
        r"""
        Remove semantically overlapped tags, keeping the most specific ones.
        """
        tag_implications = tagging.get_tag_implications()
        dan_tags = [tagging.fmt2danbooru(tag) for tag in self.tags]
        children = set()
        for tag in dan_tags:
            if child_tags := tag_implications.get(tag, None):
                children |= set(child_tags)
        self.tags = [tag for tag, dan_tag in zip(self.tags, dan_tags) if dan_tag not in children]

    def deimplicated(self):
        caption = self.copy()
        caption.deimplicate()
        return caption

    def alias(self):
        r"""
        Replace tags with their newest aliases.
        """
        tag_aliases = tagging.get_tag_aliases()
        self.tags = [(tagging.fmt2train(tag_alias) if ' ' in tag else tag_alias) if (tag_alias := tag_aliases.get(tagging.fmt2danbooru(tag), None)) else tag for tag in self.tags]

    def aliased(self):
        caption = self.copy()
        caption.alias()
        return caption

    def __iadd__(self, other):
        self.tags += Caption(other, sep=self.sep).tags
        self._empty_cache()
        return self

    def __add__(self, other):
        caption = self.copy()
        caption += other
        return caption

    def __radd__(self, other):
        return Caption(other, sep=self.sep) + self

    def __isub__(self, other):
        self.tags = [tag for tag in self.tags if tag not in Caption(other, sep=self.sep)]
        self._empty_cache()
        return self

    def __sub__(self, other):
        caption = self.copy()
        caption -= other
        return caption

    def __rsub__(self, other):
        return Caption(other, sep=self.sep) - self

    def __iand__(self, other):
        self.tags = [tag for tag in self.tags if tag in Caption(other, sep=self.sep)]
        self._empty_cache()
        return self

    def __and__(self, other):
        caption = self.copy()
        caption &= other
        return caption

    def __rand__(self, other):
        return Caption(other, sep=self.sep) & self

    def __ior__(self, other):
        self.tags = (self + other).deduplicated().tags
        self._empty_cache()
        return self

    def __or__(self, other):
        return (self + other).deduplicated()

    def __ror__(self, other):
        return Caption(other, sep=self.sep) | self

    def __reversed__(self):
        return reversed(self.tags)

    def __eq__(self, other):
        if isinstance(other, Caption):
            return self.tags == other.tags
        elif isinstance(other, (str, list)):
            return self.tags == Caption(other, sep=self.sep).tags
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.caption)

    def copy(self):
        return Caption(self.tags, sep=self.sep)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __iter__(self):
        return iter(self.tags)

    def __next__(self):
        return next(self.tags)

    def __contains__(self, tag):
        return any(tagging.match(t, tag) for t in self.tags)

    @ overload
    def __getitem__(self, index: int): ...

    @ overload
    def __getitem__(self, slice: slice): ...

    @ overload
    def __getitem__(self, tag: str): ...

    @ overload
    def __getitem__(self, pattern: re.Pattern): ...

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tags[index]
        elif isinstance(index, slice):
            return Caption(self.tags[index])
        elif isinstance(index, str):
            return self.tags[self.tags.index(index)] if index in self.tags else []
        elif isinstance(index, re.Pattern):
            return [tag for tag in self.tags if index.search(tag)]
        else:
            raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(index).__name__}'")

    @ overload
    def __setitem__(self, index: int, value: Union[str, 'Caption']): ...

    @ overload
    def __setitem__(self, slice: slice, value: Union[List[str], str, 'Caption']): ...

    @ overload
    def __setitem__(self, tag: str, value: Union[str, 'Caption']): ...

    @ overload
    def __setitem__(self, pattern: re.Pattern, value: str): ...

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if isinstance(value, str):
                self.tags[index] = value
            elif isinstance(value, Caption) and len(value) == 1:
                self.tags[index] = value.tags[0]
            else:
                raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(value).__name__}'")

        elif isinstance(index, slice):
            slice_len = len(self.tags[index])
            if isinstance(value, list) and all(isinstance(tag, str) for tag in value) and len(value) == slice_len:
                self.tags[index] = value
            elif isinstance(value, str) and len(value.split(', ')) == slice_len:
                self.tags[index] = value.split(', ')
            elif isinstance(value, Caption) and len(value) == slice_len:
                self.tags[index] = value._ags
            else:
                raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(value).__name__}'")

        elif isinstance(index, str):
            if index not in self.tags:
                return
            elif isinstance(value, str):
                self.tags[self.tags.index(index)] = value
            elif isinstance(value, Caption) and len(value) == 1:
                self.tags[self.tags.index(index)] = value.tags[0]
            else:
                raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(value).__name__}'")

        elif isinstance(index, re.Pattern):
            for i, tag in enumerate(self.tags):
                self.tags[i] = index.sub(value, tag)

    @ overload
    def __delitem__(self, index: int): ...

    @ overload
    def __delitem__(self, slice: slice): ...

    @ overload
    def __delitem__(self, tag: str): ...

    @ overload
    def __delitem__(self, pattern: re.Pattern): ...

    def __delitem__(self, index):
        if isinstance(index, int):
            del self.tags[index]
        elif isinstance(index, slice):
            del self.tags[index]
        elif isinstance(index, str):
            if index in self.tags:
                del self.tags[self.tags.index(index)]
        elif isinstance(index, re.Pattern):
            self.tags = [tag for tag in self.tags if not index.search(tag)]
        else:
            raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(index).__name__}'")

    def __len__(self):
        return len(self.tags)

    def __getattr__(self, name):
        if name in tagging.ALL_TAG_TYPES:
            cache_name = self._get_cache_name(name)
            if cache_name in self.__dict__:
                return getattr(self, cache_name)
            type_tags = get_typetags(self.tags, name)
            setattr(self, cache_name, type_tags)
            return type_tags
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in tagging.ALL_TAG_TYPES:
            cache_name = self._get_cache_name(name)
            type_tags = deduplicate([tagging.fmt2danbooru(tagging.uncomment_tag(tag)) for tag in value])
            setattr(self, cache_name, type_tags)
        else:
            super().__setattr__(name, value)

    def _get_cache_name(self, name):
        return 'cache_' + name

    def _empty_cache(self, cached_attrs=None):
        for attr in list(self.__dict__.keys()):
            if attr.startswith('cache_') and (not cached_attrs or attr in cached_attrs):
                delattr(self, attr)


def deduplicate(tags):
    res = []
    for tag in tags:
        if tag not in res:
            res.append(tag)
    return res


def get_typetags(tags, tagtype=None):
    return deduplicate([tagging.fmt2danbooru(tagging.uncomment_tag(tag)) for tag in tags if not tagtype or tagging.get_tagtype(tag) == tagtype])
