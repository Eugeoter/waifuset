import re
from typing import Union, List
from . import tagging

EMPTY_CACHE = 0


class Caption:
    r"""
    Caption object supporting concise operations.
    """

    _tags: List[str]
    _cached_properties = ('artist', 'quality', 'characters', 'styles')

    def __init__(self, caption_or_tags=None, sep=',', fix_typos: bool = True):
        if isinstance(caption_or_tags, Caption):
            tags = caption_or_tags._tags
        elif isinstance(caption_or_tags, (str, list)):
            if fix_typos and isinstance(caption_or_tags, str):
                caption_or_tags = caption_or_tags.replace('，', ',')
            tags: List[str] = tagify(caption_or_tags, sep=sep)
        elif caption_or_tags is None:
            tags: List[str] = []
        else:
            raise TypeError(f"unsupported type for caption: {type(caption_or_tags).__name__}")

        self._tags = [tag.strip() for tag in tags if tag.strip() != '']

        # caches
        self._artist: str = EMPTY_CACHE
        self._quality: str = EMPTY_CACHE
        self._characters: List[str] = EMPTY_CACHE
        self._styles: List[str] = EMPTY_CACHE

    def load_cache(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._cached_properties:
                if isinstance(value, str) and ',' in value:
                    value = [v.strip() for v in value.split(',')]
                setattr(self, f"_{key}", value)

    def clean_cache(self):
        for attr in self._cached_properties:
            setattr(self, f"_{attr}", EMPTY_CACHE)

    def copy(self):
        caption = Caption(self._tags.copy())
        cache = {key: getattr(self, f"_{key}") for key in self._cached_properties}
        cache = {key: value.copy() if isinstance(value, list) else value for key, value in cache.items()}
        caption.load_cache(**cache)
        return caption

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags):
        self._tags = tagify(tags)
        self.clean_cache()

    @property
    def caption(self):
        return captionize(self._tags)

    @caption.setter
    def caption(self, value):
        self._tags = tagify(value)

    def unique(self):
        r"""
        Caption with deduplicated tags.
        """
        return Caption(unique(self._tags))

    def unweighted(self):
        r"""
        Caption with unweighted brackets.
        """
        return Caption(tagging.REGEX_WEIGHTED_CAPTION.sub(r'\1', self.caption))

    def escaped(self):
        r"""
        Caption with escaped brackets.
        """
        return Caption(escape(self.caption))

    def unescaped(self):
        r"""
        Caption with unescaped brackets.
        """
        return Caption(unescape(self.caption))

    def replace(self, old: str, new: str, count: int = -1):
        r"""
        Caption with replaced tags.
        """
        if isinstance(old, re.Pattern):
            return self.sub(old, new)
        tags = self._tags.copy()
        for i, tag in enumerate(tags):
            tag = tag.replace(old, new)
            if count != 0 and tag != tags[i]:
                tags[i] = tag
                count -= 1
                if count == 0:
                    break
        return Caption(tags)

    def sub(self, pattern: Union[str, re.Pattern], replacement: str, count: int = -1):
        r"""
        Caption with replaced tags.
        """
        if isinstance(pattern, str):
            regex = re.compile(pattern)
        tags = self._tags.copy()
        for i, tag in enumerate(tags):
            tag = regex.sub(replacement, tag)
            if count != 0 and tag != tags[i]:
                tags[i] = tag
                count -= 1
                if count == 0:
                    break
        return Caption(tags)

    def strip(self, chars=None):
        r"""
        Caption with stripped tags.
        """
        return Caption([tag.strip(chars) for tag in self._tags if tag.strip(chars) != ''])

    def lower(self):
        r"""
        Caption with lowercased tags.
        """
        return Caption([tag.lower() for tag in self._tags])

    def underlined(self):
        return self.replace(' ', '_')

    def spaced(self):
        return self.replace('_', ' ')

    def sort(self, key=None, reverse=False):
        self._tags.sort(key=key, reverse=reverse)

    def sorted(self, key=None, reverse=False):
        return Caption(sorted(self._tags, key=key, reverse=reverse))

    def deovlped(self):
        tagging.init_overlap_table()
        caption = self.unescaped().underlined()
        table = tagging.OVERLAP_TABLE
        tags_to_remove = set()
        tag_set = set(caption._tags)
        for tag in tag_set:
            if tag in table and tag not in tags_to_remove:
                parents, children = table[tag]
                tags_to_remove |= tag_set & children
        return (caption - tags_to_remove).spaced().escaped()

    def copy(self):
        return Caption(self._tags.copy())

    def formalized(self):
        caption = self.spaced().escaped()
        for i, tag in enumerate(caption):
            if tagging.REGEX_ARTIST_TAG.match(tag):
                caption._tags[i] = f"artist: {tag[3:]}"
            elif tag in tagging.STYLE_TAGS:
                caption._tags[i] = f"style: {tag}"
            elif tagging.REGEX_CHARACTER_TAGS.match(tag):
                caption._tags[i] = f"character: {tag}"
        return Caption(caption)

    def defeatured(self, ref, threshold=0.3):
        caption = self.copy()
        if self.characters and len(self.characters) > 0:
            for char_tag in self.characters:
                freq_table = ref[char_tag]
                for tag, freq in freq_table.items():
                    if freq >= threshold:
                        caption -= tag
        return caption

    # ======================================== artist ======================================== #

    def get_artist(self):
        caption = self.caption
        match = tagging.REGEX_ARTIST.search(caption)
        if match:
            artist = match.group(3)  # update cache
        else:
            match = tagging.REGEX_ARTIST_TAG.search(caption)
            artist = match.group(2) if match else None
        self._artist = artist
        return self._artist

    @property
    def artist(self):
        if self._artist == EMPTY_CACHE:
            return self.get_artist()
        else:
            return self._artist

    @artist.setter
    def artist(self, artist):
        if artist == self.artist:
            return
        if self.artist:
            self.caption = tagging.REGEX_ARTIST.sub(rf"\2{artist}" if artist else '', self.caption)
        else:
            self._tags.insert(0, f'artist: {artist}')
        self._artist = artist

    def with_artist(self, artist):
        caption = self.copy()
        caption.artist = artist
        return caption

    # ======================================== quality ======================================== #

    def get_quality(self):
        caption = self.caption
        match = tagging.REGEX_QUALITY_TAG.search(caption)
        quality = match.group(2) if match else None
        self._quality = quality
        return self._quality

    @property
    def quality(self):
        if self._quality == EMPTY_CACHE:
            return self.get_quality()
        else:
            return self._quality

    @quality.setter
    def quality(self, quality):
        if quality == self.quality:
            return
        if self.quality:
            self.caption = tagging.REGEX_QUALITY_TAG.sub(rf"{quality}\3" if quality else '', self.caption)
        else:
            self._tags.insert(0, f"{quality} quality")
        self._quality = quality

    def with_quality(self, quality):
        caption = self.copy()
        caption.quality = quality
        return caption

    # ======================================== characters ======================================== #

    def get_characters(self):
        caption = self.caption
        characters = []
        matches = tagging.REGEX_CHARACTER.findall(caption)
        if matches:
            characters.extend([match[2] for match in matches])  # update cache
        characters.extend([tag for tag in self._tags if tagging.REGEX_CHARACTER_TAGS.match(tag)])
        if len(characters) == 0:
            characters = None
        self._characters = characters
        return self._characters

    @property
    def characters(self):
        if self._characters == EMPTY_CACHE:
            return self.get_characters()
        else:
            return self._characters

    @characters.setter
    def characters(self, characters):
        characters = tagify(characters)
        if self.characters:
            if characters == self.characters:
                return
            self._tags = [tag for tag in self._tags if not tag.startswith('character:')]
        for i in range(len(characters) - 1, -1, -1):
            self._tags.insert(0, f'character: {characters[i]}')
        self._characters = characters

    def with_characters(self, characters):
        caption = self.copy()
        caption.characters = characters
        return caption

    # ======================================== styles ======================================== #

    def get_styles(self):
        caption = self.caption
        styles = []
        matches = tagging.REGEX_STYLE.findall(caption)
        if matches:
            styles.extend([match[2] for match in matches])
        styles.extend([tag.strip() for tag in self._tags if tag in tagging.STYLE_TAGS])
        if len(styles) == 0:
            styles = None
        self._styles = styles
        return self._styles

    @property
    def styles(self):
        if self._styles == EMPTY_CACHE:
            return self.get_styles()
        else:
            return self._styles

    @styles.setter
    def styles(self, styles):
        if self.styles:
            if styles == self.styles:
                return
            self._tags = [tag for tag in self._tags if not tag.startswith('style:')]
        styles = tagify(styles)
        for i in range(len(styles) - 1, -1, -1):
            self._tags.insert(0, f'style: {styles[i]}')
        self._styles = styles

    def with_styles(self, styles):
        caption = self.copy()
        caption.styles = styles
        return caption

    def __str__(self):
        return self.caption

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return Caption(add_op(self._tags, preprocess(other)))

    def __iadd__(self, other):
        self._tags = (self + other).tags
        self.clean_cache()
        return self

    def __radd__(self, other):
        return Caption(other) + self

    def __sub__(self, other):
        return Caption(sub_op(self._tags, preprocess(other)))

    def __isub__(self, other):
        self._tags = (self - other).tags
        self.clean_cache()
        return self

    def __rsub__(self, other):
        return Caption(other) - self

    def __and__(self, other):
        return Caption(and_op(self._tags, preprocess(other)))

    def __iand__(self, other):
        self._tags = (self & other).tags
        self.clean_cache()
        return self

    def __rand__(self, other):
        return Caption(other) & self

    def __or__(self, other):
        return Caption(or_op(self._tags, preprocess(other)))

    def __ior__(self, other):
        self._tags = (self | other).tags
        self.clean_cache()
        return self

    def __ror__(self, other):
        return Caption(other) | self

    def __matmul__(self, other):
        return Caption(matmul_op(self._tags, tagify(other)))

    def __imatmul__(self, other):
        self._tags = (self @ other).tags
        return self

    def __rmatmul__(self, other):
        return Caption(other) @ self

    def __contains__(self, pattern):
        return any(match(pattern, t) for t in self._tags)

    def __len__(self):
        return len(self._tags)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._tags[index]
        elif isinstance(index, slice):
            return Caption(self._tags[index])
        else:
            raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(index).__name__}'")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if isinstance(value, str):
                self._tags[index] = value
            elif isinstance(value, Caption) and len(value) == 1:
                self._tags[index] = value._tags[0]
            else:
                raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(value).__name__}'")

        elif isinstance(index, slice):
            slice_len = len(self._tags[index])
            if isinstance(value, list) and all(isinstance(tag, str) for tag in value) and len(value) == slice_len:
                self._tags[index] = value
            elif isinstance(value, str) and len(value.split(', ')) == slice_len:
                self._tags[index] = value.split(', ')
            elif isinstance(value, Caption) and len(value) == slice_len:
                self._tags[index] = value._tags
            else:
                raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(value).__name__}'")

        self.clean_cache()

    def __delitem__(self, index):
        if isinstance(index, int):
            del self._tags[index]
        elif isinstance(index, slice):
            del self._tags[index]
        else:
            raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(index).__name__}'")

        self.clean_cache()

    def __iter__(self):
        return iter(self._tags)

    def __reversed__(self):
        return reversed(self._tags)

    def __eq__(self, other):
        if isinstance(other, Caption):
            return self._tags == other._tags
        elif isinstance(other, (str, list)):
            return self._tags == tagify(other)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.caption)


def tagify(caption_or_tags, sep=','):
    if isinstance(caption_or_tags, list):
        return caption_or_tags
    elif isinstance(caption_or_tags, str):
        return [tag.strip() for tag in caption_or_tags.split(sep)]
    elif isinstance(caption_or_tags, Caption):
        return caption_or_tags._tags
    elif caption_or_tags is None:
        return []
    else:
        raise TypeError(f"cannot convert type `{type(caption_or_tags).__name__}` to tags")


def captionize(caption_or_tags, sep=', '):
    if isinstance(caption_or_tags, list):
        return sep.join(caption_or_tags)
    elif isinstance(caption_or_tags, str):
        return caption_or_tags
    elif isinstance(caption_or_tags, Caption):
        return sep.join(caption_or_tags._tags)
    elif caption_or_tags is None:
        return ''
    else:
        raise TypeError(f"cannot convert type `{type(caption_or_tags).__name__}` to caption")


def preprocess(caption):
    if isinstance(caption, (Caption, str, list)):
        return tagify(caption)
    elif isinstance(caption, re.Pattern):
        return [caption]
    elif isinstance(caption, (set, tuple)):
        return caption
    else:
        raise TypeError(f"unsupported type for caption operations: {type(caption).__name__}")


def match(pattern, tag):
    if isinstance(pattern, str):
        return tag == pattern
    elif isinstance(pattern, re.Pattern):
        return re.match(pattern, tag)


def escape(caption):
    return re.sub(tagging.PATTERN_UNESCAPED_BRACKET, r'\\\1', caption)


def unescape(caption):
    return re.sub(tagging.PATTERN_ESCAPED_BRACKET, r'\1', caption)


def unique(self):
    res = []
    for tag in self:
        if tag not in res:
            res.append(tag)
    return res


def add_op(self, other):
    return self + other


def sub_op(self, other):
    return [t for t in self if not any(match(pat, t) for pat in other)]


def and_op(self, other):
    return [t for t in self if any(match(pat, t) for pat in other)]


def or_op(self, other):
    return add_op(self, sub_op(other, self))


def matmul_op(self, other):
    prior = []
    for pattern in other:
        level = []
        for tag in self:
            if re.match(pattern, tag):
                level.append(tag)
        if len(level) > 0:
            for tag in level:
                self.remove(tag)
            prior.append(level)
    return ', '.join([', '.join(level) for level in prior] + self)
