import os
import json
from typing import Iterable
from ..utils import log_utils as logu


def open_file_folder(path: str):
    print(f"Open {path}")
    if path is None or path == "":
        return

    command = f'explorer /select,"{path}"'
    os.system(command)


EN2CN = None
CN2EN = None


def init_cn_translation():
    global EN2CN, CN2EN
    if EN2CN is not None and CN2EN is not None:
        return
    try:
        with open('./waifuset/json/translation_cn.json', 'r', encoding='utf-8') as f:
            EN2CN = json.load(f)
        EN2CN = {k.lower(): v for k, v in EN2CN.items()}
        CN2EN = {v: k for k, v in EN2CN.items()}
    except FileNotFoundError:
        EN2CN = {}
        CN2EN = {}
        logu.warn('missing translation file: translation_cn.json')
    except Exception as e:
        EN2CN = {}
        CN2EN = {}
        logu.warn('translation_cn.json error: %s' % e)


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
