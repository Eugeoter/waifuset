import json
import time
from PIL import Image
from pathlib import Path
from typing import Tuple, Literal, Iterable
from collections import OrderedDict
from ..caption import Caption, captionize
from ...const import IMAGE_EXTS


class ImageInfo:
    # constant
    LAZY_READING = 0

    image_path: Path
    caption: Caption
    description: str
    original_size: Tuple[int, int]
    aesthetic_score: float
    safe_level: str
    safe_rating: float
    perceptual_hash: str

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
        self._image_path = Path(image_path).absolute()
        self._caption = caption if caption is None or caption is ImageInfo.LAZY_READING else Caption(caption)

        self._description = description if description is not None else None
        self._original_size = tuple(original_size) if original_size else None
        self._aesthetic_score = float(aesthetic_score) if aesthetic_score else None
        self._safe_level = str(safe_level) if safe_level else None
        self._safe_rating = float(safe_rating) if safe_rating else None
        self._perceptual_hash = str(perceptual_hash) if perceptual_hash else None

        if self._caption:
            self.caption.load_cache(**kwargs)

        # caches
        self._suffix = None
        self._stem = None
        self._category = None
        self._source = None

    def clean_cache(self, *attrs):
        for attr in attrs:
            setattr(self, f'_{attr}', None)

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = Path(value).absolute()
        self.clean_cache('suffix', 'stem', 'category', 'source', 'original_size')

    @property
    def caption(self):
        if self._caption is ImageInfo.LAZY_READING:
            self.read_txt_caption()
        return self._caption

    @caption.setter
    def caption(self, value):
        self._caption = ImageInfo.LAZY_READING if value == ImageInfo.LAZY_READING else Caption(value) if value is not None else None

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = str(value) if value is not None else None

    # @property
    # def image_size(self):
    #     if not self._image_size:
    #         with Image.open(self.image_path) as image:
    #             self._image_size = image.size
    #     return self._image_size

    @property
    def original_size(self):
        if not self.image_path.is_file() or self.image_path.suffix not in IMAGE_EXTS:
            return None
        if not self._original_size:
            with Image.open(self.image_path) as image:
                self._original_size = image.size
        return self._original_size

    @original_size.setter
    def original_size(self, value):
        self._original_size = tuple(value) if value is not None else None

    @property
    def aesthetic_score(self):
        return self._aesthetic_score

    @aesthetic_score.setter
    def aesthetic_score(self, value):
        self._aesthetic_score = float(value) if value is not None else None

    @property
    def safe_level(self):
        return self._safe_level

    @safe_level.setter
    def safe_level(self, value):
        self._safe_level = str(value) if value is not None else None

    @property
    def safe_rating(self):
        return self._safe_rating

    @safe_rating.setter
    def safe_rating(self, value):
        self._safe_rating = float(value) if value is not None else None

    @property
    def perceptual_hash(self):
        return self._perceptual_hash

    @perceptual_hash.setter
    def perceptual_hash(self, value):
        self._perceptual_hash = str(value) if value is not None else None

    @property
    def stem(self):
        if not self._stem:
            self._stem = self.image_path.stem
        return self._stem

    @property
    def key(self):
        return self.stem

    @property
    def suffix(self):
        if not self._suffix:
            self._suffix = self.image_path.suffix
        return self._suffix

    @property
    def category(self):
        if not self._category:
            self._category = self.image_path.parent.name
        return self._category

    @property
    def source(self):
        if not self._source:
            self._source = self.image_path.parent.parent
        return self._source

    @property
    def artist(self):
        return self.caption.artist if self.caption else None

    @property
    def characters(self):
        return self.caption.characters if self.caption else None

    @property
    def styles(self):
        return self.caption.styles if self.caption else None

    @property
    def quality(self):
        return self.caption.quality if self.caption else None

    def dict(self, attrs: Tuple[str] = None):
        dic = {}
        attrs = attrs or self._self_attrs
        for attr in attrs:
            dic[attr] = jsonize(getattr(self, attr))
        if 'caption' in attrs:
            for attr in self._caption_attrs:
                value = getattr(self.caption, attr) if self.caption is not None else None
                dic[attr] = ', '.join(value) if isinstance(value, list) else value
        return dic

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
        label_path = Path(label_path or self.image_path.with_suffix('.txt'))
        if label_path.is_file():
            caption = Caption(label_path.read_text(encoding='utf-8'))
            if len(caption) > 0:
                self.caption = caption

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
            elif attr == 'caption':
                self.caption.tags = value

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


ImageInfo._self_attrs = ImageInfo.__annotations__
ImageInfo._caption_attrs = Caption.__annotations__
ImageInfo._all_attrs = {**ImageInfo._self_attrs, **ImageInfo._caption_attrs}


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
    types = types or ['txt', 'danbooru']
    img_path = img_info.image_path

    txt_path = img_path.with_suffix('.txt')
    if 'txt' in types and txt_path.is_file():
        caption = txt_path.read_text(encoding='utf-8') if not lazy else ImageInfo.LAZY_READING
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
