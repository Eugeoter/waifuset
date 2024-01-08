import time
import pandas as pd
from tqdm import tqdm
from torch.utils.data import dataset as torch_dataset
from pathlib import Path
from typing import Dict, Callable
from ..data import ImageInfo
from ...utils.file_utils import listdir, smart_name
from ...const import IMAGE_EXTS
from ...utils import log_utils as logu

COLORS = [logu.ANSI.WHITE, logu.ANSI.YELLOW, logu.ANSI.CYAN, logu.ANSI.WHITE, logu.ANSI.MAGENTA, logu.ANSI.GREEN]


class Dataset(torch_dataset.Dataset):

    _data: Dict[str, ImageInfo]

    def __init__(self, source=None, condition: Callable[[ImageInfo], bool] = None, read_caption=False, formalize_caption=False, recur=True, verbose=False):
        self.verbose = verbose
        if self.verbose:
            tic = time.time()
            logu.info(f'Loading dataset...')
        dic = {}
        if not isinstance(source, (list, tuple)):
            source = [source]
        for src in source:
            if isinstance(src, (str, Path)):
                src = Path(src)
                if src.is_file():
                    suffix = src.suffix
                    if suffix in IMAGE_EXTS:  # image file
                        image_key = src.stem
                        if image_key in dic:
                            continue
                        if read_caption:
                            cap_path = src.with_suffix('.txt')
                            caption = cap_path.read_text(encoding='utf-8') if cap_path.is_file() else None
                        else:
                            caption = None
                        image_info = ImageInfo(src, caption=caption)
                        dic[image_key] = image_info

                    elif suffix == '.json':  # json file
                        import json
                        with open(src, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        for image_key, image_info in tqdm(json_data.items(), desc=f"Reading `{src.name}`", smoothing=1, disable=not verbose):
                            if image_key in dic:
                                continue
                            dic[image_key] = ImageInfo(**image_info)  # update dictionary

                    elif suffix == '.csv':  # csv file
                        df = pd.read_csv(src)
                        df = df.applymap(lambda x: None if pd.isna(x) else x)
                        for _, row in tqdm(df.iterrows(), desc=f"Reading `{src.name}`", smoothing=1, disable=not verbose):
                            image_key = row['image_key']
                            if image_key in dic:
                                continue
                            info_kwargs = {name: row[name] for name in ImageInfo._all_attrs if name in row}
                            image_info = ImageInfo(**info_kwargs)
                            dic[image_key] = image_info  # update dictionary

                    else:
                        raise ValueError(f'Invalid file source {src}.')

                elif src.is_dir():  # directory
                    files = listdir(src, exts=IMAGE_EXTS, return_path=True, return_type=Path, recur=recur)
                    for file in tqdm(files, desc=f"Reading `{src.name}`", smoothing=1, disable=not verbose):
                        image_key = file.stem
                        if image_key in dic:
                            # logu.warn(f'Duplicated image key `{image_key}`: path_1: `{dic[image_key].image_path}`, path_2: `{file}`.')
                            continue
                        if read_caption:
                            cap_path = file.with_suffix('.txt')
                            caption = cap_path.read_text(encoding='utf-8') if cap_path.is_file() else None
                        else:
                            caption = None
                        image_info = ImageInfo(file, caption=caption)
                        dic[image_key] = image_info  # update dictionary

                else:
                    raise FileNotFoundError(f'File {src} not found.')

            elif isinstance(src, ImageInfo):
                image_key = src.key
                if image_key in dic:
                    continue
                dic[image_key] = src

            elif isinstance(src, Dataset):
                for image_key, image_info in tqdm(src.items(), desc='Loading Dataset', smoothing=1, disable=not verbose):
                    if image_key in dic:
                        continue
                    dic[image_key] = image_info

            elif isinstance(src, dict):
                if src.get('image_path'):
                    image_info = ImageInfo(**src)
                    image_key = image_info.key
                    if image_key in dic:
                        continue
                    dic[image_key] = image_info

                else:
                    for image_key, image_info in tqdm(src.items(), desc='Loading dict', smoothing=1, disable=not verbose):
                        if image_key in dic:
                            continue
                        image_info = ImageInfo(**image_info)
                        dic[image_key] = image_info

            elif src is None:
                continue
            else:
                raise TypeError(f'Invalid type {type(src)} for Dataset.')

        # filter by condition
        if condition:
            dic = {image_key: image_info for image_key, image_info in dic.items() if condition(image_info)}
        # formalize caption
        if formalize_caption:
            for image_info in tqdm(dic.values(), desc='Formalizing captions', smoothing=1, disable=not verbose):
                if image_info.caption:
                    image_info.caption = image_info.caption.formalized()
        self._data = dic

        if self.verbose:
            toc = time.time()
            logu.success(f'Dataset loaded: size={len(self)}, time_cost={toc - tic:.2f}s.')

        # end init

    def make_subset(self, condition: Callable[[ImageInfo], bool] = None, *args, **kwargs):
        return Dataset(self, *args, condition=condition, **kwargs)

    def update(self, other, recur=False):
        other = Dataset(other, recur=recur)
        self._data.update(other._data)
        return self

    def pop(self, image_key):
        return self._data.pop(image_key)

    def __getitem__(self, image_key):
        return self._data[image_key]

    def __setitem__(self, image_key, image_info):
        if not isinstance(image_info, ImageInfo):
            raise TypeError('Dataset can only contain ImageInfo objects.')

        self._data[image_key] = image_info

    def __delitem__(self, image_key):
        self.pop(image_key)

    def keys(self):
        return list(self._data.keys())

    def values(self):
        return list(self._data.values())

    def items(self):
        return self._data.items()

    def df(self):
        headers = ['image_key'] + [name for name in ImageInfo._all_attrs]
        data = []
        for image_key, image_info in tqdm(self.items(), desc='Converting to DataFrame', smoothing=1, disable=not self.verbose):
            row = dict(
                image_key=image_key,
                **image_info.dict()
            )
            data.append(row)
        df = pd.DataFrame(data, columns=headers)
        return df

    def __str__(self):
        return str(self.df())

    def __repr__(self):
        return repr(self._data)

    def apply_map(self, func, *args, **kwargs):
        if self.verbose:
            tic = time.time()
            logu.info(f'Applying map `{logu.yellow(func.__name__)}`...')

        for image_info in tqdm(self.values(), desc='Applying map', smoothing=1, disable=not self.verbose):
            image_info = func(image_info, *args, **kwargs)
            assert isinstance(image_info, ImageInfo), f'Invalid return type for map `{func.__name__}`. Expected `ImageInfo`, but got `{type(image_info)}`.'

        if self.verbose:
            toc = time.time()
            logu.success(f'Map applied: time_cost={toc - tic:.2f}s.')

        return self

    def sort_keys(self):
        self._data = dict(sorted(self._data.items(), key=lambda x: x[0]))

    def stat(self):
        counter = {
            'missing_caption': [],
            'missing_category': [],
            'missing_image_file': [],
        }
        for image_key, image_info in tqdm(self.items(), desc='Stat', smoothing=1, disable=not self.verbose):
            if image_info.caption is None:
                counter['missing_caption'].append(image_key)
            if image_info.category is None:
                counter['missing_category'].append(image_key)
            if not image_info.image_path.is_file():
                counter['missing_image_file'].append(image_key)
        return counter

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, image_key):
        return image_key in self._data

    def __add__(self, other):
        if isinstance(other, Dataset):
            return Dataset(list(self._data.values()) + list(other._data.values()))
        else:
            raise TypeError(f'Invalid type {type(other)} for addition.')

    def __iadd__(self, other):
        if isinstance(other, Dataset):
            self._data.update(other._data)
            return self
        else:
            raise TypeError(f'Invalid type {type(other)} for addition.')

    def to_csv(self, fp, mode='w', sep=','):
        if self.verbose:
            tic = time.time()
            logu.info(f'Dumping dataset to `{logu.yellow(Path(fp).absolute())}`...')

        dump_as_csv(self, fp, mode=mode, sep=sep, verbose=self.verbose)

        if self.verbose:
            toc = time.time()
            logu.success(f'Dataset dumped: time_cost={toc - tic:.2f}s.')

    def to_json(self, fp, mode='w', indent=4, sort_keys=False):
        if self.verbose:
            tic = time.time()
            logu.info(f'Dumping dataset to `{logu.yellow(Path(fp).absolute())}`...')

        dump_as_json(self, fp, mode=mode, indent=indent, sort_keys=sort_keys, verbose=self.verbose)

        if self.verbose:
            toc = time.time()
            logu.success(f'Dataset dumped: time_cost={toc - tic:.2f}s.')

    def to_txts(self):
        if self.verbose:
            tic = time.time()
            logu.info(f'Dumping dataset to txt...')

        dump_as_txts(self, verbose=self.verbose)

        if self.verbose:
            toc = time.time()
            logu.success(f'Dataset dumped: time_cost={toc - tic:.2f}s.')


def dump_as_json(source, fp, mode='a', indent=4, sort_keys=False, verbose=False):
    import json
    dataset = Dataset(source)
    if mode == 'a' and Path(fp).is_file():
        dataset = Dataset(fp, verbose=verbose).update(dataset)
    json_data = {image_key: image_info.dict() for image_key, image_info in tqdm(dataset.items(), desc='Converting to dict', smoothing=1, disable=not dataset.verbose)}
    with open(fp, mode='w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=indent, sort_keys=sort_keys)


def dump_as_csv(source, fp, mode='a', sep=',', verbose=False):
    dataset = Dataset(source)
    if mode == 'a' and Path(fp).is_file():
        dataset = Dataset(fp, verbose=verbose).update(dataset)
    df = dataset.df()
    df.to_csv(fp, mode='w', sep=sep, index=False)


def dump_as_txts(source, ask_confirm=True, verbose=False):
    dataset = Dataset(source)
    if ask_confirm:
        logger = logu.FileLogger(smart_name('./logs/.tmp/%date%-%increment%.log'), name=dump_as_txts.__name__, temp=True)
        for image_key, image_info in tqdm(dataset.items(), desc='Stage 1/2: Checking', smoothing=1, disable=not verbose):
            image_path = image_info.image_path
            label_path = image_path.with_suffix('.txt')

            if not image_path.is_file():  # if image file doesn't exist
                logger.info(f"[{image_key:>20}] miss image file: {image_path}")
            if label_path.exists():
                logger.info(f"[{image_key:>20}] overwrite label file: {label_path}")
        logu.info(f"log to `{logu.yellow(logger.fp)}`")

    if not ask_confirm or input(logu.green(f"continue? ([y]/n): ")) == 'y':
        for image_key, image_info in tqdm(dataset.items(), desc='Stage 2/2: Writing to disk' if ask_confirm else 'Writing to disk', smoothing=1, disable=not verbose):
            image_info.image_path.parent.mkdir(parents=True, exist_ok=True)
            image_info.write_caption()
    else:
        if verbose:
            logu.info('Aborted.')
