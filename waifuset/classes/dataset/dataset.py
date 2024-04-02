import time
import math
import pandas as pd
import concurrent.futures as cf
from tqdm import tqdm
from pathlib import Path
from typing import List, Callable, Literal
from ..data import ImageInfo
from ...utils.file_utils import listdir, smart_name
from ...const import IMAGE_EXTS
from ...utils import log_utils as logu


class Dataset(logu.Logger):

    verbose: bool
    exts: set

    def __init__(self, source=None, key_condition: Callable[[str], bool] = None, read_attrs=False, read_types: Literal['txt', 'danbooru'] = None, lazy_reading=True, formalize_caption=False, recur=True, cacheset=None, exts=IMAGE_EXTS, verbose=False, **kwargs):
        self.init_logger(prefix_color=logu.ANSI.BRIGHT_MAGENTA)
        self.verbose = verbose
        self.exts = exts
        key_condition = key_condition or (lambda x: True)
        if not isinstance(source, (list, tuple)):
            source = [source]
        # if self.verbose:
        #     tic = time.time()
        #     self.log(f'loading dataset')

        dic = {}
        for src in source:
            if isinstance(src, (str, Path)):
                src = Path(src)
                if src.is_file():
                    suffix = src.suffix
                    if suffix in exts:  # 1. image file
                        image_key = src.stem
                        if image_key in dic or not key_condition(image_key):
                            continue
                        if cacheset and image_key in cacheset:
                            dic[image_key] = cacheset[image_key]
                            continue
                        image_info = ImageInfo(src)
                        if read_attrs:
                            image_info.read_attrs(types=read_types, lazy=lazy_reading)
                        dic[image_key] = image_info

                    elif suffix == '.json':  # 2. json file
                        import json
                        with open(src, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        for image_key, image_info in self.pbar(json_data.items(), desc=f"reading `{src.name}`", smoothing=1, disable=not verbose):
                            if image_key in dic or not key_condition(image_key):
                                continue
                            if cacheset and image_key in cacheset:
                                dic[image_key] = cacheset[image_key]
                                continue
                            dic[image_key] = ImageInfo(**image_info)  # update dictionary

                    elif suffix == '.csv':  # 3. csv file
                        df = pd.read_csv(src)
                        df = df.applymap(lambda x: None if pd.isna(x) else x)
                        for _, row in self.pbar(df.iterrows(), desc=f"reading `{src.name}`", smoothing=1, disable=not verbose):
                            image_key = row['image_key']
                            if image_key in dic or not key_condition(image_key):
                                continue
                            if cacheset and image_key in cacheset:
                                dic[image_key] = cacheset[image_key]
                                continue
                            info_kwargs = {name: row[name] for name in ImageInfo._all_attrs if name in row}
                            image_info = ImageInfo(**info_kwargs)
                            dic[image_key] = image_info  # update dictionary

                    else:
                        raise ValueError(f'Invalid file source {src}.')

                elif src.is_dir():  # 4. directory
                    files = listdir(src, exts=exts, return_path=True, return_type=Path, recur=recur)
                    for file in self.pbar(files, desc=f"reading `{src.name}`", smoothing=1, disable=not verbose):
                        image_key = file.stem
                        if image_key in dic or not key_condition(image_key):
                            # logu.warn(f'Duplicated image key `{image_key}`: path_1: `{dic[image_key].image_path}`, path_2: `{file}`.')
                            continue
                        if cacheset and image_key in cacheset:
                            dic[image_key] = cacheset[image_key]
                            continue
                        image_info = ImageInfo(file)
                        if read_attrs:
                            image_info.read_attrs(types=read_types, lazy=lazy_reading)
                        dic[image_key] = image_info  # update dictionary

                else:
                    raise FileNotFoundError(f'File {src} not found.')

            elif isinstance(src, ImageInfo):  # 5. ImageInfo object
                image_key = src.key
                if image_key in dic or not key_condition(image_key):
                    continue
                if cacheset and image_key in cacheset:
                    dic[image_key] = cacheset[image_key]
                    continue
                dic[image_key] = src

            elif isinstance(src, Dataset):  # 6. Dataset object
                for image_key, image_info in tqdm(src.items(), desc='loading Dataset', smoothing=1, disable=not verbose):
                    if image_key in dic or not key_condition(image_key):
                        continue
                    if cacheset and image_key in cacheset:
                        dic[image_key] = cacheset[image_key]
                        continue
                    dic[image_key] = image_info

            elif isinstance(src, dict):  # dict
                if src.get('image_path'):  # 7. metadata dict
                    image_info = ImageInfo(**src)
                    image_key = image_info.key
                    if image_key in dic or not key_condition(image_key):
                        continue
                    if cacheset and image_key in cacheset:
                        dic[image_key] = cacheset[image_key]
                        continue
                    dic[image_key] = image_info

                else:  # 8. img_key: img_info dict
                    for image_key, image_info in tqdm(src.items(), desc='loading dict', smoothing=1, disable=not verbose):
                        if image_key in dic or not key_condition(image_key):
                            continue
                        if cacheset and image_key in cacheset:
                            dic[image_key] = cacheset[image_key]
                            continue
                        if isinstance(image_info, dict):
                            image_info = ImageInfo(**image_info)
                        elif isinstance(image_info, ImageInfo):
                            pass
                        dic[image_key] = image_info

            elif src is None:  # 9. None
                continue
            else:
                raise TypeError(f'Invalid type {type(src)} for Dataset.')

        # formalize caption
        if formalize_caption:
            for image_info in tqdm(dic.values(), desc='formalizing captions', smoothing=1, disable=not verbose):
                if image_info.caption is not None:
                    image_info.caption = image_info.caption.formalized()
        self._data = dic

        # if self.verbose:
        #     toc = time.time()
        #     self.log(f'dataset loaded: size={len(self)}, time_cost={toc - tic:.2f}s.')

        # end init

    def make_subset(self, condition: Callable[[ImageInfo], bool] = None, cls=None, *args, **kwargs):
        import inspect
        cls = cls or self.__class__
        verbose = kwargs.get('verbose', self.verbose)
        init_params = inspect.signature(cls.__init__).parameters.keys()
        attrs_kwargs = {k: getattr(self, k) for k in cls.__annotations__ if k in init_params and k not in kwargs}
        data = {}
        for image_key, image_info in tqdm(self.items(), desc='making subset', smoothing=1, disable=not verbose):
            if condition(image_info):
                data[image_key] = image_info
        return cls(data, *args, **kwargs, **attrs_kwargs)

    def update(self, other, recur=False):
        other = Dataset(other, recur=recur)
        self._data.update(other._data)
        return self

    def pop(self, image_key, default=None):
        return self._data.pop(image_key, default)

    def clear(self):
        self._data.clear()

    def __getitem__(self, image_key):
        return self._data[image_key]

    def __setitem__(self, image_key, image_info):
        if not isinstance(image_info, ImageInfo):
            raise TypeError('Dataset can only contain ImageInfo objects.')
        self._data[image_key] = image_info

    def __delitem__(self, image_key):
        self.pop(image_key)

    def get(self, image_key, default=None):
        return self._data.get(image_key, default)

    def keys(self):
        return list(self._data.keys())

    def values(self):
        return list(self._data.values())

    def items(self):
        return self._data.items()

    def df(self):
        headers = ['image_key'] + [name for name in ImageInfo._all_attrs]
        data = []
        for image_key, image_info in self.pbar(self.items(), desc='converting DataFrame', smoothing=1, disable=not self.verbose):
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

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def apply_map(self, func, *args, max_workers=1, verbose=None, **kwargs):
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            tic = time.time()
            self.log(f'apply map `{logu.yellow(func.__name__)}`...')

        pbar = self.pbar(self.items(), desc=f'applying map `{logu.yellow(func.__name__)}`', smoothing=1, disable=not verbose)
        func_ = logu.track_tqdm(pbar)(func)
        if max_workers == 1:
            for image_key, image_info in self.items():
                self[image_key] = func_(image_info.copy(), *args, **kwargs)
        else:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(func_, image_info.copy(), *args, **kwargs) for image_info in self.values()]

                for future in cf.as_completed(futures):
                    image_info = future.result()
                    self[image_info.key] = image_info

        pbar.close()
        if verbose:
            toc = time.time()
            self.log(f'map applied: time_cost={toc - tic:.2f}s. | average={len(self) / (toc - tic):.2f} img/s.')

        return self

    def with_map(self, func, *args, max_workers=1, condition: Callable[[ImageInfo], bool] = None, read_attrs=False, read_types: Literal['txt', 'waifuc'] = None, lazy_reading=True, formalize_caption=False, recur=True, verbose=None, **kwargs):
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            tic = time.time()
            self.log(f'With map `{logu.yellow(func.__name__)}`...')

        pbar = self.pbar(self.items(), desc=f'with map `{logu.yellow(func.__name__)}`', smoothing=1, disable=not verbose)
        func_ = logu.track_tqdm(pbar)(func)
        if max_workers == 1:
            result = [func_(image_info.copy(), *args, **kwargs) for image_info in self.values()]
        else:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(func_, image_info.copy(), *args, **kwargs) for image_info in self.values()]
                result = [future.result() for future in cf.as_completed(futures)]

        pbar.close()
        if verbose:
            toc = time.time()
            self.log(f'with map applied: time_cost={toc - tic:.2f}s.')

        return Dataset(result, key_condition=condition, read_attrs=read_attrs, read_types=read_types, lazy_reading=lazy_reading, formalize_caption=formalize_caption, recur=recur, verbose=verbose)

    def sort_keys(self):
        self._data = dict(sorted(self._data.items(), key=lambda x: x[0]))

    def stat(self):
        counter = {
            'missing_caption': [],
            'missing_category': [],
            'missing_image_file': [],
        }
        for image_key, image_info in self.pbar(self.items(), desc='Stat', smoothing=1, disable=not self.verbose):
            if image_info.caption is None:
                counter['missing_caption'].append(image_key)
            if image_info.category is None:
                counter['missing_category'].append(image_key)
            if not image_info.image_path.is_file():
                counter['missing_image_file'].append(image_key)
        return counter

    def sample(self, condition=None, n=1, randomly=False, random_seed=None) -> 'Dataset':
        if randomly:
            import random
            if random_seed is not None:
                random.seed(random_seed)
            samples = random.sample([image_info for image_info in self.values() if not condition or condition(image_info)], min(n, len(self)))
        else:
            samples = [image_info for image_info in self.values() if not condition or condition(image_info)][:n]
        return Dataset(samples)

    def sort(self, key, reverse=False):
        self._data = dict(sorted(self._data.items(), key=key, reverse=reverse))

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, image_key):
        return image_key in self._data

    def to_csv(self, fp, mode='w', sep=','):
        if self.verbose:
            tic = time.time()
            self.log(f'Dumping dataset to `{logu.yellow(Path(fp).absolute())}`...')

        dump_as_csv(self, fp, mode=mode, sep=sep, verbose=self.verbose)

        if self.verbose:
            toc = time.time()
            self.log(f'Dataset dumped: time_cost={toc - tic:.2f}s.')

    def to_json(self, fp, mode='w', indent=4, sort_keys=False, verbose=None):
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            tic = time.time()
            self.log(f'Dumping dataset to `{logu.yellow(Path(fp).absolute())}`...')

        dump_as_json(self, fp, mode=mode, indent=indent, sort_keys=sort_keys, verbose=verbose)

        if verbose:
            toc = time.time()
            self.log(f'Dataset dumped: time_cost={toc - tic:.2f}s.')

    def to_txts(self):
        if self.verbose:
            tic = time.time()
            self.log(f'Dumping dataset to txt...')

        dump_as_txts(self, verbose=self.verbose)

        if self.verbose:
            toc = time.time()
            self.log(f'Dataset dumped: time_cost={toc - tic:.2f}s.')

    def split(self, *ratio, shuffle=True) -> List['Dataset']:
        keys = list(self.keys())
        if shuffle:
            import random
            random.shuffle(keys)
        datasets = []
        r_sum = sum(ratio)
        r_norm = [r / r_sum for r in ratio]
        r_cumsum = [0] + [sum(r_norm[:i + 1]) for i in range(len(r_norm))]
        for i in range(len(ratio)):
            start = int(r_cumsum[i] * len(keys))
            end = int(r_cumsum[i + 1] * len(keys))
            datasets.append(Dataset({key: self[key] for key in keys[start:end]}))
        return datasets

    def split_n(self, n: int, shuffle: bool = True) -> List['Dataset']:
        keys = list(self.keys())
        if shuffle:
            import random
            random.shuffle(keys)
        stride = math.ceil(len(keys) / n)
        datasets = [Dataset({key: self[key] for key in keys[i:i + stride]}) for i in range(0, len(keys), stride)]
        return datasets

    def batches(self, batch_size: int, shuffle: bool = True) -> List['Dataset']:
        keys = list(self.keys())
        if shuffle:
            import random
            random.shuffle(keys)
        batches = [[self[key] for key in keys[i:i + batch_size]] for i in range(0, len(keys), batch_size)]
        return batches

    def __add__(self, other):
        return Dataset({**self._data, **other._data})

    def __iadd__(self, other):
        self._data.update(other._data)
        return self

    def __and__(self, other):
        return Dataset({key: image_info for key, image_info in self.items() if key in other})

    def __iand__(self, other):
        self._data = {key: image_info for key, image_info in self.items() if key in other}
        return self

    def __or__(self, other):
        return Dataset({**other._data, **self._data})

    def __ior__(self, other):
        self._data = {**other._data, **self._data}
        return self

    def __sub__(self, other):
        return Dataset({key: image_info for key, image_info in self.items() if key not in other})

    def __isub__(self, other):
        self._data = {key: image_info for key, image_info in self.items() if key not in other}
        return self


def dump_as_json(source, fp, mode='a', indent=4, sort_keys=False, verbose=False):
    import json
    dataset = Dataset(source) if not isinstance(source, Dataset) else source
    if mode == 'a' and Path(fp).is_file():
        dataset = Dataset(fp, verbose=verbose).update(dataset)
    json_data = {}
    for image_key, image_info in tqdm(dataset.items(), desc='converting to dict', smoothing=1, disable=not dataset.verbose):
        json_data[image_key] = image_info.dict()
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
        for image_key, image_info in tqdm(dataset.items(), desc='stage 1/2: Checking', smoothing=1, disable=not verbose):
            image_path = image_info.image_path
            label_path = image_path.with_suffix('.txt')

            if not image_path.is_file():  # if image file doesn't exist
                logger.info(f"[{image_key:>20}] miss image file: {image_path}")
            if label_path.exists():
                logger.info(f"[{image_key:>20}] overwrite label file: {label_path}")
        logu.info(f"log to `{logu.yellow(logger.fp)}`")

    if not ask_confirm or input(logu.green(f"continue? ([y]/n): ")) == 'y':
        for image_key, image_info in tqdm(dataset.items(), desc='stage 2/2: Writing to disk' if ask_confirm else 'Writing to disk', smoothing=1, disable=not verbose):
            image_info.image_path.parent.mkdir(parents=True, exist_ok=True)
            image_info.write_txt_caption()
    else:
        if verbose:
            logu.info('Aborted.')
