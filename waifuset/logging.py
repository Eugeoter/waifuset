import time
import json
import copy
from typing import Literal, Dict


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


class ConsoleLogger:
    _loggers = {}
    _default_color = ANSI.BRIGHT_BLUE
    _level2color = {
        'debug': ANSI.BRIGHT_MAGENTA,
        'info': ANSI.BRIGHT_WHITE,
        'warning': ANSI.BRIGHT_YELLOW,
        'error': ANSI.BRIGHT_RED,
        'critical': ANSI.BRIGHT_RED,
    }
    DISABLE = False

    def __new__(cls, name, *args, **kwargs):
        if name not in cls._loggers:
            cls._loggers[name] = super().__new__(cls)
        return cls._loggers[name]

    def __init__(self, name, prefix_msg=None, prefix_color=None, disable: bool = False):
        self.name = name
        self.prefix_msg = prefix_msg or name
        self.color = color2ansi(prefix_color) if isinstance(prefix_color, str) else (prefix_color or self._default_color)
        self.disable = disable

    def get_disable(self) -> bool:
        return ConsoleLogger.DISABLE or self.disable

    def set_disable(self, disable: bool):
        self.disable = disable

    def get_prefix(self, level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info', prefix_msg=None, prefix_color=None) -> str:
        prefixes = []
        level_color = self._level2color.get(level, "")
        if level != 'info':
            level_str = f"({stylize(level, level_color)})"
            prefixes.append(level_str)
        prefix_msg = prefix_msg or self.prefix_msg
        prefix_color = prefix_color or self.color
        if prefix_msg:
            prefix_str = f"[{stylize(prefix_msg, prefix_color)}]"
            prefixes.append(prefix_str)
        return ' '.join(prefixes)

    def print(self, *msg: str, level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info', prefix_msg=None, prefix_color=None, no_prefix=False, disable=None, write=False, **kwargs):
        disable = disable if disable is not None else self.get_disable()
        if not disable:
            if not no_prefix:
                prefix = self.get_prefix(level, prefix_msg, prefix_color)
                msg = (prefix,) + msg
            if write:
                from tqdm import tqdm
                sep = kwargs.pop('sep', ' ')
                end = kwargs.pop('end', '\n')
                s = sep.join(map(str, msg))
                tqdm.write(s, end=end)
            else:
                print(*msg, **kwargs)

    def tqdm(self, *args, level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info', prefix_msg=None, prefix_color=None, no_prefix=False, **kwargs):
        from tqdm import tqdm
        desc = []
        if not no_prefix:
            prefix = self.get_prefix(level, prefix_msg, prefix_color)
            desc.append(prefix)
        if 'desc' in kwargs:
            desc.append(kwargs['desc'])
        kwargs["desc"] = ' '.join(desc)
        kwargs["disable"] = kwargs.get('disable', self.get_disable())
        return tqdm(*args, **kwargs)

    def timer(self, name=None, level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info', **kwargs):
        return timer(name, level, logger=self, **kwargs)

    def info(self, *msg: str, **kwargs):
        self.print(*msg, level='info', **kwargs)

    def debug(self, *msg: str, **kwargs):
        self.print(*msg, level='debug', **kwargs)

    def warning(self, *msg: str, **kwargs):
        self.print(*msg, level='warning', **kwargs)

    def error(self, *msg: str, **kwargs):
        self.print(*msg, level='error', **kwargs)

    def critical(self, *msg: str, **kwargs):
        self.print(*msg, level='critical', **kwargs)

    def __deepcopy__(self, memo):
        new_instance = self.__class__(
            name=self.name,
            prefix_msg=self.prefix_msg,
            prefix_color=self.color,
            disable=self.disable,
        )
        return new_instance


def get_logger(name, prefix_msg=None, prefix_color=None, disable: bool = False) -> ConsoleLogger:
    r"""
    Get or create a logger with the specified name.
    """
    return ConsoleLogger(name, prefix_msg, prefix_color, disable)


def getLogger(name, prefix_msg=None, prefix_color=None, disable: bool = False) -> ConsoleLogger:
    r"""
    Get or create a logger with the specified name.

    Compatibility with the standard logging module.
    """
    return get_logger(name, prefix_msg, prefix_color, disable)


def get_all_loggers() -> Dict[str, ConsoleLogger]:
    return ConsoleLogger._loggers


def set_all_loggers_disable(disable: bool):
    ConsoleLogger.DISABLE = disable
    for logger in get_all_loggers().values():
        logger.disable = disable


logger = get_logger('logging')


class timer:
    def __init__(self, name=None, level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info', logger=None):
        self.name = name
        self.level = level
        self.logger = logger or get_logger('timer')
        assert isinstance(self.logger, ConsoleLogger), f"logger must be an instance of ConsoleLogger, but got {type(self.logger)}"

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        prefix = f"{self.name}: " if self.name else ""
        self.logger.print(prefix + f"{end_time - self.start_time:.2f}s", level=self.level)

    def __call__(self, func):
        def inner(*args, **kwargs):
            with timer(self.name, level=self.level, logger=self.logger):
                return func(*args, **kwargs)
        return inner


def track_tqdm(pbar, n: int = 1):
    r"""
    Track the progress of a tqdm progress bar. Once the decorated function is called, the progress bar will be updated by n steps.
    """
    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            finally:
                pbar.update(n)
            return res
        return inner
    return wrapper


def time_test(func, *args, n: int = 1, **kwargs):
    r"""
    Time the execution of a function.
    """
    import time
    start_time = time.time()
    for _ in range(n):
        func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time


# ANSI color tools


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


def color2ansi(color: str) -> str:
    return getattr(ANSI, color.upper(), "")


def red(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_RED, format_spec=format_spec, newline=newline)


def green(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_GREEN, format_spec=format_spec, newline=newline)


def yellow(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_YELLOW, format_spec=format_spec, newline=newline)


def blue(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_BLUE, format_spec=format_spec, newline=newline)


def magenta(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_MAGENTA, format_spec=format_spec, newline=newline)


def cyan(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_CYAN, format_spec=format_spec, newline=newline)


def white(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_WHITE, format_spec=format_spec, newline=newline)


def black(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BRIGHT_BLACK, format_spec=format_spec, newline=newline)


def bold(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.BOLD, format_spec=format_spec, newline=newline)


def underline(msg: str, format_spec: str = "", newline: bool = False) -> str:
    return stylize(msg, ANSI.UNDERLINE, format_spec=format_spec, newline=newline)

# Standard logging tools


def info(*msg: str, **kwargs):
    logger.info(*msg, **kwargs)


def debug(*msg: str, **kwargs):
    logger.debug(*msg, **kwargs)


def warning(*msg: str, **kwargs):
    logger.warning(*msg, **kwargs)


def error(*msg: str, **kwargs):
    logger.error(*msg, **kwargs)


def critical(*msg: str, **kwargs):
    logger.critical(*msg, **kwargs)


def tqdm(*args, **kwargs):
    return logger.tqdm(*args, **kwargs)

# Special logging tools


def title(msg: str = "", sep: str = "=") -> str:
    r"""
    Get a title string with a centered message.

    Example:
    >>> title("Hello, world!")
    ====================== Hello, world! ======================

    >>> title("Hello, world!", sep="-")
    ---------------------- Hello, world! ----------------------
    """
    import shutil
    width = shutil.get_terminal_size().columns
    if msg:
        total_sep_len = (width - len(msg) - 2) // 2
        centered_msg = sep * total_sep_len + ' ' + msg + ' ' + sep * total_sep_len
        if len(centered_msg) < width:
            centered_msg += sep
    else:
        centered_msg = sep * width
    return centered_msg


def jsonize(obj, ensure_ascii=False, indent=4, sort_keys=False) -> str:
    r"""
    Serialize a data-structure to a json-like string.
    """
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
