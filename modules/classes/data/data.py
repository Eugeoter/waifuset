from PIL import Image
from pathlib import Path
from typing import Tuple
from ..caption import Caption, captionize


class ImageInfo:
    LAZY_READING = 0

    image_path: Path
    caption: Caption
    original_size: Tuple[int, int]

    def __init__(
        self,
        image_path,
        caption=None,
        original_size=None,
        **kwargs,
    ):
        self._image_path = Path(image_path).absolute()
        self._caption = caption if caption is None or caption is ImageInfo.LAZY_READING else Caption(caption)
        self._original_size = tuple(original_size) if original_size else None

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
            self.read_caption()
        return self._caption

    @caption.setter
    def caption(self, value):
        self._caption = Caption(value) if value is not None else None

    # @property
    # def image_size(self):
    #     if not self._image_size:
    #         with Image.open(self.image_path) as image:
    #             self._image_size = image.size
    #     return self._image_size

    @property
    def original_size(self):
        if not self._original_size:
            with Image.open(self.image_path) as image:
                self._original_size = image.size
        return self._original_size

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

    def dict(self):
        dic = {}
        dic['image_path'] = self.image_path.absolute().as_posix()
        dic['caption'] = str(self.caption) if self.caption else None
        dic['original_size'] = list(self.original_size)
        for attr_key in self._caption_attrs:
            attr_value = getattr(self.caption, attr_key) if self.caption else None
            if attr_value:
                attr_value = captionize(attr_value)
            dic[attr_key] = attr_value
        return dic

    # def dict(self):
    #     dic = {}
    #     dic['image_path'] = self.image_path
    #     dic['caption'] = self.caption
    #     for attr_key in self._caption_attrs:
    #         attr_value = getattr(self.caption, attr_key) if self.caption else None
    #         dic[attr_key] = attr_value
    #     return dic

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
            if value:
                hash_str += str(value)
        return hash(hash_str)

    def read_caption(self, label_path=None):
        label_path = Path(label_path or self.image_path.with_suffix('.txt'))
        if label_path.is_file():
            caption = Caption(label_path.read_text(encoding='utf-8'))
            if len(caption) > 0:
                self.caption = caption

    def write_caption(self, label_path=None):
        if not self.caption:
            return
        label_path = Path(label_path or self.image_path.with_suffix('.txt'))
        label_path.write_text(str(self.caption) if self.caption else '', encoding='utf-8')

    def copy(self):
        return ImageInfo(
            image_path=self.image_path,
            caption=self.caption.copy() if self.caption else None,
            original_size=self.original_size,
        )

    _all_attrs = ('image_path', 'caption', 'artist', 'styles', 'quality', 'characters', 'original_size')
    _self_attrs = ('image_path', 'caption', 'original_size')
    _caption_attrs = ('artist', 'styles', 'quality', 'characters')
