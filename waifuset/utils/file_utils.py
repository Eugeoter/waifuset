import os
import time
import re
from pathlib import Path
from typing import Optional, Iterable
from ..const import StrPath


def listdir(
    directory: StrPath,
    exts: Optional[Iterable[str]] = None,
    return_type: Optional[type] = None,
    return_path: Optional[bool] = False,
    return_dir: Optional[bool] = False,
    recur: Optional[bool] = False,
    return_abspath: Optional[bool] = True,
):
    r"""
    List files in a directory.
    :param directory: The directory to list files in.
    :param exts: The extensions to filter by. If None, all files are returned.
    :param return_type: The type to return the files as. If None, returns the type of the directory. If return_path is True, returns str anyway.
    :param return_path: Whether to return the full path of the files.
    :param return_dir: Whether to return directories instead of files.
    :param recur: Whether to recursively list files in subdirectories.
    :param return_abspath: Whether to return absolute paths.
    :return: A list of files in the directory.
    """
    if exts and return_dir:
        raise ValueError("Cannot return both files and directories")

    if not return_path and return_type and return_type != str:
        raise ValueError("Cannot return non-str type when returning name")

    if not recur:
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
    else:
        files = []
        for root, dirs, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.join(root, f))

    if exts:
        files = [f for f in files if os.path.splitext(f)[1] in exts]
    if return_dir:
        files = [f for f in files if os.path.isdir(f)]
    if not return_path:
        files = [os.path.basename(f) for f in files]
    if return_abspath:
        files = [os.path.abspath(f) for f in files]
    if return_type == Path:
        files = [return_type(f) for f in files]

    return files


def smart_name(
    filename_pattern: str,
    increment_extensions: Optional[Iterable[str]] = None,
):
    r"""
    Replace the following placeholders in the filename:
        - %datetime%: current time in the format of %Y-%m-%d-%H-%M-%S
        - %date%: current date in the format of %Y-%m-%d
        - %time%: current time in the format of %H-%M-%S
        - %increment%: incrementing number in the basename, starting from 0, iterating until the filename is unique under the increment_extensions.
    :param filename_pattern: The filename_pattern to replace.
    """
    return_type = type(filename_pattern)
    if isinstance(filename_pattern, Path):
        filename_pattern = str(filename_pattern)
    filename_pattern = filename_pattern.replace('%datetime%', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    filename_pattern = filename_pattern.replace('%date%', time.strftime("%Y-%m-%d", time.localtime()))
    filename_pattern = filename_pattern.replace('%time%', time.strftime("%H-%M-%S", time.localtime()))

    if '%increment%' in os.path.basename(filename_pattern):
        increment_extensions = increment_extensions or Path(filename_pattern).suffix
        basename = os.path.basename(filename_pattern)
        all_stems = [os.path.splitext(p)[0] for p in os.listdir(os.path.dirname(filename_pattern))
                     if os.path.splitext(p)[1] in increment_extensions] if os.path.isdir(os.path.dirname(filename_pattern)) else []
        i = 0
        filestem = os.path.splitext(basename)[0].replace('%increment%', str(i))

        while filestem in all_stems:
            i += 1
            filestem = os.path.splitext(basename)[0].replace('%increment%', str(i))

        filename_pattern = str(Path(filename_pattern).with_name(f"{filestem}{Path(filename_pattern).suffix}"))

    assert re.search(r'%.*%', filename_pattern) is None, "Invalid filename pattern: {}".format(filename_pattern)

    return return_type(filename_pattern)


def smart_path(root, name, exts: Optional[Iterable[str]] = tuple(), return_type: Optional[type] = None):
    return_type = return_type or type(name)
    if isinstance(name, Path):
        name = str(name)
    name = name.replace('%datetime%', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    name = name.replace('%date%', time.strftime("%Y-%m-%d", time.localtime()))
    name = name.replace('%time%', time.strftime("%H-%M-%S", time.localtime()))

    if '%index%' in name:
        ext_names = [Path(name).with_suffix(ext) for ext in exts]
        idx = 0
        while os.path.exists(path := os.path.join(root, name.replace('%index%', str(idx)))) or any(os.path.exists(os.path.join(root, ext_name.replace('%increment%', str(idx)))) for ext_name in ext_names):
            idx += 1
    else:
        path = os.path.join(root, name)

    return return_type(path)


def remove_empty(root: StrPath, recur: Optional[bool] = False):
    r"""
    Remove empty directories in the given directory.
    """
    for dir_p in listdir(root, return_path=True, return_dir=True, recur=recur, return_type=str)[::-1]:
        if len(os.listdir(dir_p)) == 0:
            os.rmdir(dir_p)


def formalize_name(s):
    from googletrans import Translator
    # 1. split s into chinese, japanese, koran and english parts
    pattern = re.compile(r'([\u4e00-\u9fa5]+)|([\u3040-\u309f\u30a0-\u30ff]+)|([\uac00-\ud7a3]+)|([\w]+)')
    # 2. translate chinese, japanese, koran parts into english and replace them in s
    s = pattern.sub(lambda m: Translator().translate(m.group(0), dest='en').text if not m.group(0).isascii() else m.group(0), s)
    # 3. remove all non-ascii characters
    s = re.sub(r'[^\x00-\x7f]', r'', s)
    return s


def download_from_url(url, cache_dir=None, verbose=True):
    from huggingface_hub import hf_hub_download
    split = url.split("/")
    if len(split) >= 3:
        username, repo_id, model_name = split[-3], split[-2], split[-1]
    if verbose:
        print(f"[download_from_url]: {username}/{repo_id}/{model_name}")
    model_path = hf_hub_download(f"{username}/{repo_id}", model_name, cache_dir=cache_dir)
    return model_path
