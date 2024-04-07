import re
from typing import Union, List, Literal
from . import tagging

LAZY_READING = 999
LAZY_LOADING = 998


class Caption:
    r"""
    Caption object supporting concise operations.
    """

    # tags: List[str]
    artist: str
    quality: str
    characters: list
    styles: list

    # if caption_or_tags is a Caption object, return caption_or_tags itself
    def __new__(cls, caption_or_tags=None, sep=', ', fix_typos: bool = True):
        if isinstance(caption_or_tags, Caption):
            return caption_or_tags
        else:
            return super(Caption, cls).__new__(cls)

    def __init__(self, caption_or_tags=None, sep=', ', fix_typos: bool = True):
        if isinstance(caption_or_tags, Caption):
            self = caption_or_tags
            return
        elif isinstance(caption_or_tags, (str, list)):
            if fix_typos and isinstance(caption_or_tags, str):
                caption_or_tags = caption_or_tags.replace('，', ',')
            tags: List[str] = tagify(caption_or_tags, sep=sep)
        elif caption_or_tags is None:
            tags: List[str] = []
        else:
            raise TypeError(f"unsupported type for caption: {type(caption_or_tags).__name__}")

        self._sep = sep  # separator
        self._tags = [tag.strip() for tag in tags if tag.strip() != '']

        # caches
        self._artist: str = LAZY_LOADING
        self._quality: str = LAZY_LOADING
        self._characters: List[str] = LAZY_LOADING
        self._styles: List[str] = LAZY_LOADING

    def load_cache(self, **kwargs):
        r"""
        Directly set the cache of properties.
        """
        for key, value in kwargs.items():
            if key in self._cached_properties:
                if value is not None and self._cached_properties[key] == list:
                    value = [v.strip() for v in value.split(',')]
                setattr(self, f"_{key}", value)

    def clean_cache(self):
        r"""
        Reset all cached properties to `LAZY_LOADING`.
        """
        for attr in self._cached_properties:
            setattr(self, f"_{attr}", LAZY_LOADING)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags):
        self._tags = tagify(tags, sep=self._sep)
        if self.characters:
            for ch_tag in self.characters:
                if ch_tag not in self._tags and f"character: {ch_tag}" not in self._tags:
                    self._characters.remove(ch_tag)
            if not self.characters:
                self._characters = LAZY_LOADING
        if self.styles:
            for style in self.styles:
                if style not in self._tags and f"style: {style}" not in self._tags:
                    self._styles.remove(style)
            if not self.styles:
                self._styles = LAZY_LOADING
        if self.artist:
            if self.artist not in self._tags and f"artist: {self.artist}" not in self._tags:
                self._artist = LAZY_LOADING
        if self.quality:
            if self.quality not in self._tags:
                self._quality = LAZY_LOADING

    @property
    def caption(self):
        return captionize(self._tags, sep=self._sep)

    @caption.setter
    def caption(self, value):
        self.tags = tagify(value, sep=self._sep)

    def unique(self):
        r"""
        Caption with deduplicated tags.
        """
        return Caption(unique(self.tags))

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
        tags = self.tags.copy()
        for i, tag in enumerate(tags):
            tag = tag.replace(old, new)
            if count != 0 and tag != tags[i]:
                tags[i] = tag
                count -= 1
                if count == 0:
                    break
        return Caption(tags)

    def replace_tag(self, old: str, new: str, count: int = -1):
        r"""
        Caption with replaced tags.
        """
        tags = self.tags.copy()
        for i, tag in enumerate(tags):
            if match(old, tag):
                if isinstance(old, str):
                    tags[i] = new
                else:
                    tags[i] = old.sub(new, tag)
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
        else:
            regex = pattern
        tags = self.tags.copy()
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
        return Caption([tag.strip(chars) for tag in self.tags if tag.strip(chars) != ''])

    def lower(self):
        r"""
        Caption with lowercased tags.
        """
        return Caption([tag.lower() for tag in self.tags])

    def underlined(self):
        r"""
        Caption with spaces replaced by underscores.
        """
        return self.replace(' ', '_')

    def spaced(self):
        r"""
        Caption with underscores replaced by spaces.
        """
        return self.replace('_', ' ')

    def sort(self, key=None, reverse=False):
        r"""
        Sort tags.
        """
        key = key or tagging.tag2priority
        self.tags.sort(key=key, reverse=reverse)

    def sorted(self, key=None, reverse=False):
        return Caption(sorted(self.tags, key=key, reverse=reverse))

    def deoverlap(self):
        r"""
        Remove semantically overlapped tags, keeping the most specific ones.
        """
        tagging.init_overlap_table()
        dan2tag = {fmt2danbooru(tag): tag for tag in self.tags}
        tag2dan = {v: k for k, v in dan2tag.items()}
        ovlp_table = tagging.OVERLAP_TABLE
        tags_to_remove = set()
        tagset = set(self.tags)
        for tag in tagset:
            dantag = tag2dan[tag]
            if dantag in ovlp_table and tag not in tags_to_remove:
                parents, children = ovlp_table[dantag]
                parents = {dan2tag[parent] for parent in parents if parent in dan2tag}
                children = {dan2tag[child] for child in children if child in dan2tag}
                tags_to_remove |= tagset & children
        self._tags = [tag for tag in self.tags if tag not in tags_to_remove]  # deoverlap won't change properties

    def deoverlaped(self):
        caption = self.copy()
        caption.deoverlap()
        return caption

    def parse(self):
        r"""
        According to the danbooru wiki, extract artist, characters, and styles tags.
        """
        metatags = self.get_metatags()
        if (artist_tag := metatags['artist']) and not artist_tag.startswith('artist:'):
            self._tags[self._tags.index(artist_tag)] = f"artist: {artist_tag}"
        if (characters_tags := metatags['characters']):
            for character_tag in characters_tags:
                if not character_tag.startswith('character:'):
                    self._tags[self._tags.index(character_tag)] = f"character: {character_tag}"
        if (styles_tags := metatags['styles']):
            for style_tag in styles_tags:
                if not style_tag.startswith('style:'):
                    self._tags[self._tags.index(style_tag)] = f"style: {style_tag}"

    def parsed(self):
        caption = self.copy()
        caption.parse()
        return caption

    def formalize(self):
        r"""
        Add prefixes to meta tags.
        """
        self.tags = [formalize(tag) for tag in self.tags]

    def formalized(self):
        caption = self.copy()
        caption.formalize()
        return caption

    def deformalize(self):
        r"""
        Remove prefixes from meta tags.
        """
        self.tags = [remove_prefix(tag, by_artist=True) for tag in self.tags]

    def deformalized(self):
        caption = self.copy()
        caption.deformalize()
        return caption

    def defeature(self, feature_table=None, **kwargs):
        r"""
        According to the feature table which is extracted from danbooru wiki, remove feature tags of every characters.
        """
        if not self.characters:
            return
        if not feature_table:
            tagging.init_feature_table(**kwargs)
            feature_table = tagging.FEATURE_TABLE
        all_features = set()
        for character in self.characters:
            features = feature_table.get(character, None)
            if features:
                all_features |= features
        self._tags = [tag for tag in self.tags if fmt2standard(tag) not in all_features]  # defeature won't change properties

    def defeatured(self, feature_table=None, **kwargs):
        caption = self.copy()
        caption.defeature(feature_table, **kwargs)
        return caption

    def demeta(self, tagtype: Literal['artist', 'character', 'style', 'quality', 'copyright']):
        tagset = tagging.get_tagset(tagtype)
        self.tags = [tag for tag in self.tags if fmt2danbooru(tag) not in tagset]

    def demetaed(self, tagtype: Literal['artist', 'character', 'style', 'quality', 'copyright']):
        caption = self.copy()
        caption.demeta(tagtype)
        return caption

    def get_metatags(self):
        special_tags = get_metatags(self.tags, copyrights=False)
        artist, characters, styles, quality = special_tags['artist'], special_tags['characters'], special_tags['styles'], special_tags['quality']
        self._artist = remove_prefix(artist) if artist else None
        self._characters = [remove_prefix(tag) for tag in characters] if characters else None
        self._styles = [remove_prefix(tag) for tag in styles] if styles else None
        self._quality = quality
        return special_tags

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
        if self._artist == LAZY_LOADING:
            return self.get_artist()
        else:
            return self._artist

    @artist.setter
    def artist(self, artist):
        if artist == self.artist:
            return
        if self.artist:
            for i, tag in enumerate(self.tags):
                if tag2type(tag) == 'artist':
                    self.tags[i] = tag.replace(self.artist, artist)
                    break
        else:
            self.tags.insert(0, f"artist: {artist}")
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
        if self._quality == LAZY_LOADING:
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
            self.tags.insert(0, f"{quality} quality")
        self._quality = quality

    def with_quality(self, quality):
        caption = self.copy()
        caption.quality = quality
        return caption

    # ======================================== characters ======================================== #

    def get_characters(self):
        characters = [remove_prefix(tag) for tag in self.tags if tag2type(tag) == 'character']
        self._characters = characters if characters else None
        return self._characters

    @property
    def characters(self):
        if self._characters == LAZY_LOADING:
            return self.get_characters()
        else:
            return self._characters

    @characters.setter
    def characters(self, characters):
        # characters = tagify(characters)
        # if self.characters:
        #     if characters == self.characters:
        #         return
        #     self.tags = [tag for tag in self.tags if not tag.startswith('character:')]
        # for i in range(len(characters) - 1, -1, -1):
        #     self.tags.insert(0, f'character: {characters[i]}')
        # self._characters = characters
        raise NotImplementedError("characters setter is not implemented")

    def with_characters(self, characters):
        caption = self.copy()
        caption.characters = characters
        return caption

    # ======================================== styles ======================================== #

    def get_styles(self):
        tagging.init_custom_tags()  # init style tags
        caption = self.caption
        styles = []
        matches = tagging.REGEX_STYLE.findall(caption)
        if matches:
            styles.extend([match[2] for match in matches])
        styles.extend([tag.strip() for tag in self.tags if tag in tagging.STYLE_TAGS])
        if len(styles) == 0:
            styles = None
        self._styles = styles
        return self._styles

    @property
    def styles(self):
        if self._styles == LAZY_LOADING:
            return self.get_styles()
        else:
            return self._styles

    @styles.setter
    def styles(self, styles):
        # if styles == self.styles:
        #     return
        # elif styles is None:
        #     self.tags = [tag for tag in self.tags if tag2type(tag) != 'style']
        #     self._styles = None
        #     return
        # else:
        #     for i, tag in enumerate(self.tags):
        #         fmt_tag = fmt2standard(tag)
        #         for style in styles:
        #             if fmt_tag == fmt2standard(style):
        #                 styles.remove(style)
        #                 self.tags[i] = tag.replace(fmt_tag, style)
        #                 if not styles:
        #                     break
        #     if styles:
        #         for i in range(len(styles) - 1, -1, -1):
        #             self.tags.insert(0, f"style: {styles[i]}")
        #         self._styles = LAZY_LOADING
        raise NotImplementedError("styles setter is not implemented")

    def with_styles(self, styles):
        caption = self.copy()
        caption.styles = styles
        return caption

    @property
    def copyrights(self):
        return [tag for tag in self.tags if fmt2danbooru(tag) in tagging.COPYRIGHT_TAGS] if tagging.init_copyright_tags() else None

    def attr_dict(self):
        return {
            'artist': self.artist,
            'quality': self.quality,
            'characters': captionize(self.characters) if self.characters else None,
            'styles': captionize(self.styles) if self.styles else None,
        }

    def __str__(self):
        return self.caption

    def __repr__(self):
        return self.__str__()

    def dict(self):
        return {
            'caption': str(self.caption),
            **self.attr_dict(),
        }

    def df(self):
        import pandas as pd
        return pd.DataFrame([self.dict()])

    def __add__(self, other):
        return Caption(add_op(self.tags, preprocess_caption(other, sep=self._sep)))

    def __iadd__(self, other):
        self.tags = (self + other).tags
        self.clean_cache()
        return self

    def __radd__(self, other):
        return Caption(other) + self

    def __sub__(self, other):
        return Caption(sub_op(self.tags, preprocess_caption(other, sep=self._sep)))

    def __isub__(self, other):
        self.tags = (self - other).tags
        self.clean_cache()
        return self

    def __rsub__(self, other):
        return Caption(other) - self

    def __and__(self, other):
        return Caption(and_op(self.tags, preprocess_caption(other, sep=self._sep)))

    def __iand__(self, other):
        self.tags = (self & other).tags
        self.clean_cache()
        return self

    def __rand__(self, other):
        return Caption(other) & self

    def __or__(self, other):
        return Caption(or_op(self.tags, preprocess_caption(other, sep=self._sep)))

    def __ior__(self, other):
        self.tags = (self | other).tags
        self.clean_cache()
        return self

    def __ror__(self, other):
        return Caption(other) | self

    def __matmul__(self, other):
        return Caption(matmul_op(self.tags, tagify(other, sep=self._sep), sep=self._sep))

    def __imatmul__(self, other):
        self.tags = (self @ other).tags
        return self

    def __rmatmul__(self, other):
        return Caption(other) @ self

    def __contains__(self, pattern):
        return any(match(pattern, t) for t in self.tags)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tags[index]
        elif isinstance(index, slice):
            return Caption(self.tags[index])
        else:
            raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(index).__name__}'")

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

        self.clean_cache()

    def __delitem__(self, index):
        if isinstance(index, int):
            del self.tags[index]
        elif isinstance(index, slice):
            del self.tags[index]
        else:
            raise TypeError(f"unsupported operand type(s) for []: 'Caption' and '{type(index).__name__}'")

        self.clean_cache()

    def __iter__(self):
        return iter(self.tags)

    def __reversed__(self):
        return reversed(self.tags)

    def __eq__(self, other):
        if isinstance(other, Caption):
            return self.tags == other.tags
        elif isinstance(other, (str, list)):
            return self.tags == tagify(other)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.caption)


Caption._cached_properties = Caption.__annotations__


def tag2type(tag: str):
    if ':' in tag:
        if tag.startswith('artist:'):
            return 'artist'
        elif tag.startswith('character:'):
            return 'character'
        elif tag.startswith('style:'):
            return 'style'
        elif tag.startswith('quality:'):
            return 'quality'
        elif tag.startswith('copyright:'):
            return 'copyright'
    elif 'quality' in tag:
        return 'quality'
    elif tag.startswith('by ') and tagging.REGEX_ARTIST_TAG.match(tag):
        return 'artist'

    dan_tag = fmt2danbooru(tag)
    if tagging.init_artist_tags() and dan_tag in tagging.ARTIST_TAGS:
        return 'artist'
    elif tagging.init_character_tags() and dan_tag in tagging.CHARACTER_TAGS:
        return 'character'
    elif tagging.init_custom_tags() and dan_tag in tagging.STYLE_TAGS:
        return 'style'
    elif tagging.init_copyright_tags() and dan_tag in tagging.COPYRIGHT_TAGS:
        return 'copyright'
    elif dan_tag in tagging.QUALITY_TAGS:
        return 'quality'
    else:
        return 'general'


def formalize(tag):
    tagtype = tag2type(tag)
    if tagtype in ('artist', 'character', 'style'):
        attr = tag.split(':')[-1].strip('_ ')
        return f"{tagtype}: {attr}"
    else:
        return tag


def get_metatags(tags, artist=True, characters=True, styles=True, quality=True, copyrights=True):
    dic = {}
    if artist:
        dic['artist'] = None
        tagging.init_artist_tags()
    if characters:
        dic['characters'] = None
        tagging.init_character_tags()
    if styles:
        dic['styles'] = None
        tagging.init_custom_tags()
    if quality:
        dic['quality'] = None
        tagging.init_custom_tags()
    if copyrights:
        dic['copyrights'] = None
        tagging.init_copyright_tags()

    for tag in tags:
        ftag = fmt2danbooru(tag)
        if artist and ftag in tagging.ARTIST_TAGS:
            dic['artist'] = tag
        elif characters and ftag in tagging.CHARACTER_TAGS:
            dic['characters'] = dic['characters'] or []
            dic['characters'].append(tag)
        elif styles and ftag in tagging.STYLE_TAGS:
            dic['styles'] = dic['styles'] or []
            dic['styles'].append(tag)
        elif quality and ftag in tagging.QUALITY_TAGS:
            dic['quality'] = tag
        elif copyrights and ftag in tagging.COPYRIGHT_TAGS:
            dic['copyrights'] = dic['copyrights'] or []
            dic['copyrights'].append(tag)
    return dic


def tagify(caption_or_tags, sep=','):
    if isinstance(caption_or_tags, list):
        return caption_or_tags
    elif isinstance(caption_or_tags, str):
        return [tag.strip() for tag in caption_or_tags.split(sep)]
    elif isinstance(caption_or_tags, Caption):
        return caption_or_tags.tags
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
        return sep.join(caption_or_tags.tags)
    elif caption_or_tags is None:
        return ''
    else:
        raise TypeError(f"cannot convert type `{type(caption_or_tags).__name__}` to caption")


def remove_prefix(tag, by_artist=False):
    if ':' in tag:
        if tag.startswith('artist:'):
            tag = tag[7:].strip('_ ')
            if by_artist:
                tag = f"by {tag}"
            return tag
        elif tag.startswith('character:'):
            return tag[10:].strip('_ ')
        elif tag.startswith('style:'):
            return tag[6:].strip('_ ')
    return tag


def fmt2danbooru(tag):
    r"""
    Process a tag to:
    - lower case
    - replace spaces with underscores
    - unescape brackets
    - remove prefixes
    """
    tag = tag.lower().replace(' ', '_').strip('_').replace(':_', ':')
    tag = unescape(tag)
    tag = remove_prefix(tag)
    return tag


def fmt2standard(tag, by_artist=False):
    r"""
    Process a tag to:
    - lower case
    - remove underscores
    - escape brackets
    - remove prefixes
    """
    tag = tag.lower().replace('_', ' ').strip(' ')
    tag = escape(tag)
    tag = remove_prefix(tag, by_artist=by_artist)
    return tag


def preprocess_caption(caption, sep=','):
    if isinstance(caption, (Caption, str, list)):
        return tagify(caption, sep=sep)
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


def escape(s):
    return re.sub(tagging.PATTERN_UNESCAPED_BRACKET, r'\\\1', s)


def unescape(s):
    return re.sub(tagging.PATTERN_ESCAPED_BRACKET, r'\1', s)


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


def matmul_op(self, other, sep=','):
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
    return sep.join([sep.join(level) for level in prior] + self)
