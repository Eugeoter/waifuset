import json
from PIL import Image
from pathlib import Path
from typing import Tuple, Literal, Any
from ..caption import Caption, captionize


class ImageInfo:
    LAZY_READING = 0

    image_path: Path
    caption: Caption
    nl_caption: str
    original_size: Tuple[int, int]
    aesthetic_score: float
    perceptual_hash: str

    def __init__(
        self,
        image_path,
        caption=None,
        nl_caption=None,
        original_size=None,
        aesthetic_score=None,
        perceptual_hash=None,
        **kwargs,
    ):
        self._image_path = Path(image_path).absolute()
        self._caption = caption if caption is None or caption is ImageInfo.LAZY_READING else Caption(caption)
        self._nl_caption = nl_caption if nl_caption is not None else None
        self._original_size = tuple(original_size) if original_size else None
        self._aesthetic_score = float(aesthetic_score) if aesthetic_score else None
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
    def nl_caption(self):
        return self._nl_caption

    @nl_caption.setter
    def nl_caption(self, value):
        self._nl_caption = str(value) if value is not None else None

    # @property
    # def image_size(self):
    #     if not self._image_size:
    #         with Image.open(self.image_path) as image:
    #             self._image_size = image.size
    #     return self._image_size

    @property
    def original_size(self):
        if not self.image_path.is_file():
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

    def dict(self, attrs: Tuple[Literal['image_path', 'caption', 'nl_caption', 'original_size', 'aesthetic_score', 'perceptual_hash']] = None):
        dic = {}
        attrs = set(attrs or self._self_attrs)
        if 'image_path' in attrs:
            dic['image_path'] = self.image_path.absolute().as_posix()
        if 'caption' in attrs:
            dic['caption'] = str(self.caption) if self.caption else None
        if 'nl_caption' in attrs:
            dic['nl_caption'] = self.nl_caption
        if 'original_size' in attrs:
            dic['original_size'] = list(self.original_size) if self.original_size else None
        if 'aesthetic_score' in attrs:
            # keep 3 digits
            dic['aesthetic_score'] = round(self.aesthetic_score, 3) if self.aesthetic_score is not None else None
        if 'perceptual_hash' in attrs:
            dic['perceptual_hash'] = self.perceptual_hash
        if 'caption' in attrs:
            for attr_key in self._caption_attrs:
                attr_value = getattr(self.caption, attr_key) if self.caption else None
                if attr_value:
                    attr_value = captionize(attr_value)
                dic[attr_key] = attr_value
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

    def read_attrs(self, types: Literal['txt', 'waifuc'] = None, lazy=False):
        attrs = get_attrs(self, types=types, lazy=lazy)
        for attr, value in attrs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)

    def copy(self):
        return ImageInfo(
            image_path=self.image_path,
            caption=self.caption.copy() if self.caption is not None else None,
            nl_caption=self.nl_caption,
            original_size=self.original_size,
            aesthetic_score=self.aesthetic_score,
            perceptual_hash=self.perceptual_hash,
        )

    _all_attrs = ('image_path', 'caption', 'nl_caption', 'quality', 'artist', 'styles', 'characters', 'original_size', 'aesthetic_score', 'perceptual_hash')
    _self_attrs = ('image_path', 'caption', 'nl_caption', 'original_size', 'aesthetic_score', 'perceptual_hash')
    _caption_attrs = ('quality', 'artist', 'styles', 'characters')


def parse_waifuc_metadata(metadata):
    metadata = metadata['danbooru']
    tags = metadata['tag_string'].split(' ')
    artist_tags = metadata['tag_string_artist'].split(' ')
    character_tags = metadata['tag_string_character'].split(' ')

    for artist in artist_tags:
        tags = [f"artist:{artist}" if tag == artist else tag for tag in tags]
    for character in character_tags:
        tags = [f"character:{character}" if tag == character else tag for tag in tags]

    caption = ', '.join([tag.replace('_', ' ') for tag in tags])
    artist_tags = [tag.replace(' ', '_') for tag in artist_tags]
    character_tags = [tag.replace(' ', '_') for tag in character_tags]

    return {
        'caption': caption,
        'artist': artist_tags,
        'characters': character_tags,
        'original_size': (metadata['image_width'], metadata['image_height']),
    }


def get_attrs(img_info: ImageInfo, types: Literal['txt', 'waifuc'] = None, lazy=False):
    if isinstance(types, str):
        types = [types]
    types = types or ['txt', 'waifuc']
    img_path = img_info.image_path
    if 'txt' in types and (txt_path := img_path.with_suffix('.txt')).is_file():
        caption = txt_path.read_text(encoding='utf-8') if not lazy else ImageInfo.LAZY_READING
        attrs = {
            'image_path': img_path,
            'caption': caption,
        }
    elif 'waifuc' in types and (waifuc_md_path := img_path.with_name(f".{img_path.stem}_meta.json")).is_file():
        with open(waifuc_md_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        attrs = parse_waifuc_metadata(metadata)
    else:
        attrs = {}
    return attrs
