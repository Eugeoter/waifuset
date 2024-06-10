import logging
import re
import copy
import time
from pathlib import Path
from tqdm import tqdm
from typing import Iterable


def track_tqdm(pbar: tqdm, n: int = 1):
    def wrapper(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            pbar.update(n)
            return res
        return inner
    return wrapper


def stylize(string: str, *ansi_styles, format_spec: str = "", newline: bool = False) -> str:
    r"""
    Stylize a string by a list of ANSI styles.
    """
    if not isinstance(string, str):
        string = format(string, format_spec)
    if len(ansi_styles) == 0:
        return string
    ansi_styles = ''.join(ansi_styles)
    if newline:
        ansi_styles = ANSI.NEWLINE + ansi_styles
    return ansi_styles + string + ANSI.RESET


class ANSI:
    BLACK = '\033[30m'  # basic colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'  # bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    F = '\033[F'
    K = '\033[K'
    NEWLINE = F + K


def camel_to_snake(name):
    # 将所有连续的大写字母转换为小写，但在最后一个大写字母前加下划线
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)

    # 将剩余的大写字母转换为小写，同时在它们前面加上下划线
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def color2ansi(color: str):
    return getattr(ANSI, color.upper(), "")


class ConsoleLogger:
    _loggers = {}
    _default_color = "bright_blue"

    def __new__(cls, name, *args, **kwargs):
        if name not in cls._loggers:
            cls._loggers[name] = super().__new__(cls)
        return cls._loggers[name]

    def __init__(self, name, prefix_str=None, color=None, disable: bool = False):
        self.name = name
        self.prefix_str = prefix_str or name
        self.colortype = color
        self.color = color2ansi(color or self._default_color)
        self.disable = disable

    @property
    def prefix(self):
        return f"[{stylize(self.prefix_str, self.color)}]" if self.prefix_str else ""

    def print(self, *msg: str, no_prefix=False, disable=None, **kwargs):
        disable = disable if disable is not None else self.disable
        if not disable:
            if not no_prefix:
                msg = (self.prefix,) + msg
            print(*msg, **kwargs)

    def tqdm(self, *args, no_prefix=False, **kwargs):
        from tqdm import tqdm
        kwargs["desc"] = f"{self.prefix}{kwargs.get('desc', '')}"
        kwargs["disable"] = kwargs.get('disable', self.disable)
        return tqdm(*args, **kwargs)

    def __deepcopy__(self, memo):
        new_instance = self.__class__(
            name=self.name,
            prefix_str=self.prefix_str,
            color=self.colortype,
            disable=self.disable
        )
        return new_instance


def get_logger(name, prefix_str=None, color=None, disable: bool = False):
    return ConsoleLogger(name, prefix_str, color, disable)


def get_all_loggers():
    return ConsoleLogger._loggers


debug_logger = get_logger('debug', color='bright_yellow')


def info(msg: str, *args, **kwargs):
    print('[INFO] ' + stylize(msg), *args, **kwargs)


def warn(msg: str, *args, **kwargs):
    print('[WARNING] ' + stylize(msg, ANSI.YELLOW), *args, **kwargs)


def error(msg: str, *args, **kwargs):
    print('[ERROR] ' + stylize(msg, ANSI.BRIGHT_RED), *args, **kwargs)


def success(msg: str, *args, **kwargs):
    print('[INFO] ' + stylize(msg, ANSI.BRIGHT_GREEN), *args, **kwargs)


def red(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_RED, format_spec=format_spec, newline=newline)


def green(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_GREEN, format_spec=format_spec, newline=newline)


def yellow(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_YELLOW, format_spec=format_spec, newline=newline)


def blue(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_BLUE, format_spec=format_spec, newline=newline)


def magenta(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_MAGENTA, format_spec=format_spec, newline=newline)


def cyan(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_CYAN, format_spec=format_spec, newline=newline)


def white(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_WHITE, format_spec=format_spec, newline=newline)


def black(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BRIGHT_BLACK, format_spec=format_spec, newline=newline)


def bold(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.BOLD, format_spec=format_spec, newline=newline)


def underline(msg: str, format_spec: str = "", newline: bool = False):
    return stylize(msg, ANSI.UNDERLINE, format_spec=format_spec, newline=newline)


class FileLogger:
    def __init__(self, fp: str, level: int = logging.INFO, name: str = None, disable=False, temp=False):
        self.fp = Path(fp).absolute()
        self.fp.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(fp)
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger
        self.file_handler = file_handler
        self.formatter = formatter
        self.temp = temp
        self.disable = disable

    def __del__(self):
        self.file_handler.close()
        self.logger.removeHandler(self.file_handler)
        if self.temp and self.fp.is_file():
            self.fp.unlink()

    def info(self, msg: str):
        if not self.disable:
            self.logger.info(msg)

    def debug(self, msg: str):
        if not self.disable:
            self.logger.debug(msg)

    def warn(self, msg: str):
        if not self.disable:
            self.logger.warning(msg)

    def error(self, msg: str):
        if not self.disable:
            self.logger.error(msg)


class timer:
    def __init__(self, name, logger=debug_logger):
        self.func_name = name
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        self.logger.print(f"`{self.func_name}`: {end_time - self.start_time:.2f}s")

    def __call__(self, func):
        def inner(*args, **kwargs):
            with timer(self.func_name, self.logger):
                return func(*args, **kwargs)
        return inner
