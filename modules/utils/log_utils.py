import logging
from pathlib import Path
from tqdm import tqdm


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


def debug(msg: str, *args, **kwargs):
    print('[DEBUG] ' + stylize(msg, ANSI.BOLD, ANSI.BRIGHT_WHITE), *args, **kwargs)


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
