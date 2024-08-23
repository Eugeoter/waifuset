import os
import gradio as gr
import json
from pathlib import Path
from argparse import Namespace
from functools import wraps
from typing import Union, Tuple, Literal, Iterable, Dict, Callable, overload
from waifuset import logging, tagging

ROOT = Path(__file__).parent.parent

logger = logging.get_logger('UI')

COL2TYPE_BASE = {
    'image_key': str,
    'image_path': str,
    'description': str,
}

COL2TYPE_EXTRA = {
    'source': str,
    'category': str,
    'original_size': str,
    'perceptual_hash': str,
    'aesthetic_score': float,
    'safe_rating': float,
    'date': str,
}

COL2TYPE_CAPTION = {
    'caption': str,
    **{tagtype: str for tagtype in tagging.TAG_TYPES}
}

COL2TYPE = {
    **COL2TYPE_BASE,
    **COL2TYPE_EXTRA,
    **COL2TYPE_CAPTION,
}

SORTING_METHODS: Dict[str, Callable] = {
}

FORMAT_PRESETS = {
    'train': tagging.fmt2train,
    'prompt': tagging.fmt2prompt,
    'danbooru': tagging.fmt2danbooru,
    'awa': tagging.fmt2awa,
    'unescape': tagging.fmt2unescape,
    'escape': tagging.fmt2escape,
}


class UIState(Namespace):
    pass


class UITab:
    def __init__(self, tab: gr.Tab):
        self.tab = tab


class UIBuffer(object):
    def __init__(self):
        self.buffer = {}

    def get(self, key):
        return self.buffer.get(key, None)

    def keys(self):
        return self.buffer.keys()

    def do(self, key, value):
        self.buffer.setdefault(key, ([], []))
        self.buffer[key][0].append(value.copy())
        self.buffer[key][1].clear()

    def delete(self, key):
        if key not in self.buffer:
            return None
        del self.buffer[key]

    def undo(self, key):
        if key not in self.buffer or len(self.buffer[key][0]) == 1:
            return None
        self.buffer[key][1].append(self.buffer[key][0].pop())
        return self.buffer[key][0][-1]

    def redo(self, key):
        if key not in self.buffer or not self.buffer[key][1]:
            return None
        self.buffer[key][0].append(self.buffer[key][1].pop())
        return self.buffer[key][0][-1]

    def latest(self, key):
        if key not in self.buffer or len(self.buffer[key][0]) <= 1:
            return None
        return self.buffer[key][0][-1]

    def latests(self):
        latests = {key: self.latest(key) for key in self.keys()}
        latests = {key: value for key, value in latests.items() if value is not None}
        return latests

    def __contains__(self, key):
        return key in self.buffer

    def __len__(self):
        return len(self.buffer)


class UIGallerySelectData:
    def __init__(self, index=None, key=None):
        self.index = index
        self.key = key

    @overload
    def select(self, selected: gr.SelectData): ...

    @overload
    def select(self, selected: Tuple[int, str]): ...

    def select(self, selected: Union[gr.SelectData, Tuple[int, str]]):
        if isinstance(selected, gr.SelectData):
            self.index = selected.index
            image_filename = selected.value['image']['orig_name']
            img_key = os.path.basename(os.path.splitext(image_filename)[0])
            self.key = img_key
        elif isinstance(selected, tuple):
            self.index, self.key = selected
        elif selected is None:
            self.index, self.key = None, None
        else:
            raise NotImplementedError


def EmojiButton(value, variant: Literal['primary', 'secondary', 'stop'] = "secondary", scale=0, min_width=40, *args, **kwargs):
    return gr.Button(value=value, variant=variant, scale=scale, min_width=min_width, *args, **kwargs)


def track_progress(progress: gr.Progress, desc=None, total=None, n=1):
    progress.n = 0  # initial call
    progress(0, desc=desc, total=total)

    def wrapper(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            progress.n += n
            progress((progress.n, total), desc=desc, total=total)
            return res
        return inner
    return wrapper


def open_file_folder(path: str):
    print(f"Open {path}")
    if path is None or path == "":
        return

    command = f'explorer /select,"{path}"'
    os.system(command)


def search_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None


EN2CN = None
CN2EN = None
TRANSLATION_TABLE_PATH = search_file('translation_cn.json', ROOT)
if TRANSLATION_TABLE_PATH is None:
    logger.warning(f'missing translation file under {ROOT}: translation_cn.json')


def init_cn_translation():
    global EN2CN, CN2EN
    if EN2CN is not None and CN2EN is not None:
        return
    try:
        with open(TRANSLATION_TABLE_PATH, 'r', encoding='utf-8') as f:
            EN2CN = json.load(f)
        EN2CN = {k.lower(): v for k, v in EN2CN.items()}
        CN2EN = {v: k for k, v in EN2CN.items()}
    except FileNotFoundError:
        EN2CN = {}
        CN2EN = {}
        logging.warn(f'missing translation file: {TRANSLATION_TABLE_PATH}')
    except Exception as e:
        EN2CN = {}
        CN2EN = {}
        logging.warn('translation_cn.json error: %s' % e)


def en2cn(text):
    if text is None:
        return None
    init_cn_translation()
    return EN2CN.get(text.lower(), text)


def cn2en(text):
    if text is None:
        return None
    init_cn_translation()
    return CN2EN.get(text, text)


def translate(text, language='en'):
    translator = {
        'en': cn2en,
        'cn': en2cn,
    }
    translator = translator.get(language, cn2en)
    if isinstance(text, str):
        return translator(text.replace('_', ' '))
    elif isinstance(text, Iterable):
        return [translator(t.replace('_', ' ')) for t in text]
    else:
        raise TypeError(f'Unsupported type: {type(text)}')


def kwargs_setter(func, **preset_kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.update(preset_kwargs)
        return func(*args, **kwargs)
    return wrapper


def patch_image_path_base_info(img_md):
    if not (img_path := img_md.get('image_path')):
        return None
    img_path = Path(img_path)
    if 'image_key' not in img_md:
        img_md['image_key'] = img_path.stem
    if 'category' not in img_md:
        img_md['category'] = img_path.parent.name
    if 'source' not in img_md:
        img_md['source'] = img_path.parent.parent.name
    return img_md


def patch_dirset(img_md):
    img_md = patch_image_path_base_info(img_md)
    img_md['caption'] = None
    return img_md
