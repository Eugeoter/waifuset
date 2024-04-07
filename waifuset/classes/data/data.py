import json
import time
from PIL import Image
from pathlib import Path
from typing import Tuple, Literal, Iterable
from collections import OrderedDict
from ..caption import Caption, captionize
from ...const import IMAGE_EXTS

LAZY_READING = 999
LAZY_LOADING = 998


def auto_convert(obj, type, precision=3):
    if obj is None:
        return None
    if isinstance(obj, list) and type is str:
        obj = ', '.join(obj)
    elif not isinstance(obj, type):
        obj = type(obj)
        if isinstance(obj, float):
            obj = round(obj, precision)
        elif isinstance(obj, Path):
            obj = obj.absolute()
    return obj


class ImageInfo:
    image_path: Path
    caption: Caption
    description: str
    original_size: Tuple[int, int]
    aesthetic_score: float
    safe_level: str
    safe_rating: float
    perceptual_hash: str

    __dicttype__ = {
        'image_path': str,
        'caption': str,
        'description': str,
        'original_size': tuple,
        'aesthetic_score': float,
        'safe_level': str,
        'safe_rating': float,
        'perceptual_hash': str,
        'artist': str,
        'characters': str,
        'styles': str,
        'quality': str,
    }

    def __init__(
        self,
        image_path,
        caption=None,
        description=None,
        original_size=None,
        aesthetic_score=None,
        safe_level=None,
        safe_rating=None,
        perceptual_hash=None,
        **kwargs,
    ):
        # main structure
        self._dict = {
            'image_path': auto_convert(image_path, str),
            'caption': auto_convert(caption, str),
            'description': auto_convert(description, str),
            'original_size': auto_convert(original_size, tuple),
            'aesthetic_score': auto_convert(aesthetic_score, float),
            'safe_level': auto_convert(safe_level, str),
            'safe_rating': auto_convert(safe_rating, float),
            'perceptual_hash': auto_convert(perceptual_hash, str),
        }

        # attr caches
        self._image_path = image_path if isinstance(image_path, Path) else LAZY_LOADING
        self._caption = caption if isinstance(caption, Caption) else LAZY_LOADING

        self._description = description if isinstance(description, str) else LAZY_LOADING
        self._original_size = original_size if isinstance(original_size, tuple) else LAZY_LOADING
        self._aesthetic_score = aesthetic_score if isinstance(aesthetic_score, float) else LAZY_LOADING
        self._safe_level = safe_level if isinstance(safe_level, str) else LAZY_LOADING
        self._safe_rating = safe_rating if isinstance(safe_rating, float) else LAZY_LOADING
        self._perceptual_hash = perceptual_hash if isinstance(perceptual_hash, str) else LAZY_LOADING

        # load caption caches
        if caption is not None and kwargs:
            self.caption.load_cache(**kwargs)

    def clean_cache(self, *attrs):
        for attr in attrs:
            setattr(self, f'_{attr}', None)

    @property
    def image_path(self):
        if self._image_path is LAZY_LOADING:
            self._image_path = auto_convert(self._dict['image_path'], Path)
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        self._dict['image_path'] = value  # set main dict
        self._image_path = LAZY_LOADING  # clean cache
        self.clean_cache('suffix', 'stem', 'category', 'source')

    @property
    def label_path(self):
        return self.image_path.with_suffix('.txt')

    @property
    def caption(self):
        if self._caption is LAZY_LOADING:
            self._caption = auto_convert(self._dict['caption'], Caption)
        if self._caption is LAZY_READING:
            self._caption = auto_convert(read_txt_caption(self.image_path.with_suffix('.txt')), Caption)
            # if self._caption is not None:
            #     self._caption.load_cache(artist=self._dict.get('artist'), characters=self._dict.get('characters'), styles=self._dict.get('styles'), quality=self._dict.get('quality'))
            self._dict['caption'] = self._caption
        return self._caption

    @caption.setter
    def caption(self, value):
        if value == LAZY_READING:
            self._caption = LAZY_READING
            return
        self._dict['caption'] = value
        self._dict['artist'] = LAZY_LOADING
        self._dict['characters'] = LAZY_LOADING
        self._dict['styles'] = LAZY_LOADING
        self._dict['quality'] = LAZY_LOADING
        self._caption = LAZY_LOADING

    @property
    def description(self):
        if self._description is LAZY_LOADING:
            self._description = auto_convert(self._dict['description'], str)
        return self._description

    @description.setter
    def description(self, value):
        self._dict['description'] = value
        self._description = LAZY_LOADING

    @property
    def original_size(self):
        if self._original_size is LAZY_LOADING:
            self._original_size = auto_convert(self._dict['original_size'], tuple)
        if self._original_size is None:
            try:
                with Image.open(self.image_path) as image:
                    self._original_size = auto_convert(image.size, tuple)
            except Exception as e:
                return None
        return self._original_size

    @original_size.setter
    def original_size(self, value):
        self._dict['original_size'] = value
        self._original_size = LAZY_LOADING

    @property
    def aesthetic_score(self):
        if self._aesthetic_score is LAZY_LOADING:
            self._aesthetic_score = auto_convert(self._dict['aesthetic_score'], float)
        return self._aesthetic_score

    @aesthetic_score.setter
    def aesthetic_score(self, value):
        self._dict['aesthetic_score'] = value
        self._aesthetic_score = LAZY_LOADING

    @property
    def safe_level(self):
        if self._safe_level is LAZY_LOADING:
            self._safe_level = auto_convert(self._dict['safe_level'], str)
        return self._safe_level

    @safe_level.setter
    def safe_level(self, value):
        self._dict['safe_level'] = value
        self._safe_level = LAZY_LOADING

    @property
    def safe_rating(self):
        if self._safe_rating is LAZY_LOADING:
            self._safe_rating = auto_convert(self._dict['safe_rating'], float)
        return self._safe_rating

    @safe_rating.setter
    def safe_rating(self, value):
        self._dict['safe_rating'] = value
        self._safe_rating = LAZY_LOADING

    @property
    def perceptual_hash(self):
        if self._perceptual_hash is LAZY_LOADING:
            self._perceptual_hash = auto_convert(self._dict['perceptual_hash'], str)
        return self._perceptual_hash

    @perceptual_hash.setter
    def perceptual_hash(self, value):
        self._dict['perceptual_hash'] = value
        self._perceptual_hash = LAZY_LOADING

    @property
    def copyrights(self):
        return self.caption.copyrights if self.caption is not None else None

    @property
    def stem(self):
        return self.image_path.stem

    @property
    def key(self):
        return self.stem

    @property
    def suffix(self):
        return self.image_path.suffix

    @property
    def category(self):
        return self.image_path.parent.name

    @property
    def source(self):
        return self.image_path.parent.parent

    @property
    def root(self):
        return self.image_path.parent.parent.parent

    @property
    def artist(self):
        return self.caption.artist if self.caption is not None else None

    @artist.setter
    def artist(self, value):
        self.caption.artist = value

    @property
    def characters(self):
        return self.caption.characters if self.caption is not None else None

    @characters.setter
    def characters(self, value):
        self.caption.characters = value

    @property
    def styles(self):
        return self.caption.styles if self.caption is not None else None

    @styles.setter
    def styles(self, value):
        self.caption.styles = value

    @property
    def quality(self):
        return self.caption.quality if self.caption is not None else None

    @quality.setter
    def quality(self, value):
        self.caption.quality = value

    def dict(self, attrs: Tuple[str] = None):
        self._dict = {k: auto_convert(v, self.__dicttype__[k]) for k, v in self._dict.items()}
        self._dict.update(self.caption.attr_dict() if self.caption is not None else {'artist': None, 'characters': None, 'styles': None, 'quality': None})
        return self._dict if attrs is None else {attr: self._dict[attr] for attr in attrs}

    def __eq__(self, other):
        if not isinstance(other, ImageInfo):
            return False
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return str(self.image_path.absolute().as_posix())

    def __hash__(self):
        hash_str = ''
        for attr in self._self_attrs:
            value = getattr(self, attr)
            if value is not None:
                hash_str += str(value)
        return hash(hash_str)

    @property
    def metadata(self):
        return Image.open(self.image_path).info

    @property
    def gen_info(self):
        from ...utils.image_utils import parse_gen_info
        return parse_gen_info(self.metadata)

    def read_txt_caption(self, label_path=None):
        self.caption = read_txt_caption(label_path or self.image_path.with_suffix('.txt'))

    def write_txt_caption(self, label_path=None):
        if not self.caption:
            return
        label_path = Path(label_path or self.image_path.with_suffix('.txt'))
        label_path.write_text(str(self.caption) if self.caption else '', encoding='utf-8')

    def read_attrs(self, types: Literal['txt', 'danbooru'] = None, lazy=True):
        try:
            attrs = read_attrs(self, types=types, lazy=lazy)
        except Exception as e:
            print(f"failed to read attrs for {self.image_path}: {e}")
            return None
        if not attrs:
            return
        for attr, value in attrs.items():
            if attr in self._self_attrs:
                setattr(self, attr, value)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def __getitem__(self, key):
        return getattr(self, key)


ImageInfo._self_attrs = ImageInfo.__annotations__
ImageInfo._caption_attrs = Caption.__annotations__
ImageInfo._all_attrs = {**ImageInfo._self_attrs, **ImageInfo._caption_attrs}


def read_txt_caption(fp):
    if (fp := auto_convert(fp, Path)).is_file():
        caption = fp.read_text(encoding='utf-8')
        if len(caption) > 0:
            return caption
    return None


def jsonize(obj):
    if obj is None:
        return None
    elif isinstance(obj, (int, str)):
        return obj
    elif isinstance(obj, float):
        return round(obj, 3)
    elif isinstance(obj, Caption):
        return str(obj)
    elif isinstance(obj, Path):
        return obj.absolute().as_posix()
    elif isinstance(obj, Iterable):
        return [jsonize(item) for item in obj]
    return obj


def parse_danbooru_metadata(metadata):
    tags = metadata['tag_string'].split(' ')
    artist_tags = metadata['tag_string_artist'].split(' ')
    character_tags = metadata['tag_string_character'].split(' ')
    meta_tags = set(metadata['tag_string_meta'].split(' '))
    safe_level = metadata['rating']

    tags = set(tag for tag in tags if tag not in meta_tags)

    for artist in artist_tags:
        tags = [f"artist: {artist}" if tag == artist else tag for tag in tags]
    for character in character_tags:
        tags = [f"character: {character}" if tag == character else tag for tag in tags]

    tags = [tag.replace('_', ' ') for tag in tags]
    artist_tags = [tag.replace('_', ' ') for tag in artist_tags]
    character_tags = [tag.replace('_', ' ') for tag in character_tags]

    return {
        'caption': tags,
        'artist': artist_tags[0] if len(artist_tags) > 0 else None,
        'characters': character_tags,
        'original_size': (metadata['image_width'], metadata['image_height']),
        'safe_level': safe_level,
    }


def read_attrs(img_info: ImageInfo, types: Literal['txt', 'danbooru'] = None, lazy=False):
    if isinstance(types, str):
        types = [types]
    types = types or ('txt', 'danbooru')
    img_path = img_info.image_path

    txt_path = img_path.with_suffix('.txt')
    if 'txt' in types and txt_path.is_file():
        caption = txt_path.read_text(encoding='utf-8') if not lazy else LAZY_READING
        attrs = {
            'image_path': img_path,
            'caption': caption,
        }
        return attrs

    if 'danbooru' in types:
        if (waifuc_md_path := img_path.with_name(f".{img_path.stem}_meta.json")).is_file():  # waifuc naming format
            with open(waifuc_md_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            attrs = parse_danbooru_metadata(metadata['danbooru'])
            return attrs
        elif (gallery_dl_md_path := img_path.with_name(f"{img_path.name}.json")).is_file():
            with open(gallery_dl_md_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            attrs = parse_danbooru_metadata(metadata)
            return attrs

    return None
