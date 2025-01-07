import requests
import os
import contextlib
from pathlib import Path
from typing import Dict, List, Any, Literal, Union, Tuple, Iterable
from .auto_dataset import AutoDataset
from .dict_dataset import DictDataset
from .sqlite3_dataset import SQLite3Dataset
from .dataset import Dataset
from ..data.huggingface_data import HuggingFaceData
from ...const import IMAGE_EXTS
from ... import mapping, logging

logger = logging.get_logger("dataset")

DEFAULT_KWARGS = {
    'dataset_type': None,
    'verbose': True,
    'column_mapping': None,
    'remove_columns': None,
    'mapping': None,
    'priority': 0,

    'tbname': 'metadata',
    'primary_key': 'image_key',
    'fp_key': 'image_path',
    'exts': IMAGE_EXTS,
    'recur': True,
    'read_attrs': False,
    'cache_dir': None,
    'token': None,
    'split': 'train',
    'max_retries': None,

    'annotations': ['instances', 'captions'],

    'additional_metadata_column': 'meta_info',
}


class FastDataset(AutoDataset):
    def __new__(cls, *source, dataset_cls: type = None, merge_mode: Literal['union', 'intersection', 'update'] = 'union', **default_kwargs) -> Dataset:
        source = parse_source_input(source)
        check_parsed_source(source)
        if merge_mode == 'chain':
            from .chain_dataset import ChainDataset
            return ChainDataset(*source, dataset_cls=dataset_cls, merge_mode=merge_mode, **default_kwargs)
        else:
            return load_fast_dataset(*source, dataset_cls=dataset_cls, merge_mode=merge_mode, **default_kwargs)

    def __init__(self, *source, dataset_cls: type = None, merge_mode: Literal['union', 'intersection', 'update'] = 'union', **default_kwargs) -> Dataset:
        r"""
        Initialize the dataset from a specified source.

        @param source: The source of the dataset(s). This can be one or multiple sources, each of which can be of the following types:
            - str or Path: The name or path to a local dataset (e.g., image directory, CSV, JSON, SQLite3) or a HuggingFace dataset (repository ID).
            - dict: The configuration for a local dataset or a HuggingFace dataset, with the following keys:
                - name_or_path: The name or path to a local dataset or a HuggingFace dataset.
                - primary_key: The primary key of the dataset. Default is 'image_key'.
                - column_mapping: The mapping of the dataset columns.
                - match_columns: Indicates whether to match the dataset columns. If True, columns not in the column_mapping will be removed.
                - fp_key: The key for the file path in the dataset. Default is 'image_path'. Applicable only for local datasets.
                - exts: The extensions of the images. Default is common image extensions ('jpg', 'jpeg', 'png', 'webp', 'jfif'). Applicable only for local datasets.
                - recur: Specifies whether to recursively search for images in the directory. Default is True. Applicable only for local datasets.
                - tbname: The table name of the SQLite3 dataset. Default is None, which means the dataset is assumed to be in a single table format. Applicable only for SQLite3 datasets.
            - Dataset: An instance of the Dataset class.

        @param dataset_cls: An optional class to use for the dataset. If not provided, a default dataset class will be used.
        @param merge_mode: The mode to merge multiple datasets. Default is 'union'.
        @param skip_missing: Whether to skip missing images. Default is False.
        @param **default_kwargs: Additional keyword arguments to pass to the dataset constructor.

        @returns: An instance of the Dataset class.
        """

    @staticmethod
    def dump(dataset, fp, *args, **kwargs):
        kwargs = {**DEFAULT_KWARGS, **kwargs}
        if kwargs['tbname'] is None:
            kwargs['tbname'] = 'metadata'
        return AutoDataset.dump(dataset, fp, *args, **kwargs)


def load_fast_dataset(
    *source: List[Union[str, Dict[str, Any], Dataset]],
    merge_mode: Literal['union', 'intersection', 'update', 'no'] = 'union',
    dataset_cls: type = None,
    **default_kwargs,
) -> Union[Dataset, List[Dataset]]:
    r"""
    Load dataset(s) from the specified source(s) and merge them into a single dataset.

    @param source: The source of the dataset(s). This can be one or multiple sources, each of which can be of the following types:
        - str or Path: The name or path to a local dataset (e.g., image directory, CSV, JSON, SQLite3) or a HuggingFace dataset (repository ID).
        - dict: The configuration for a local dataset or a HuggingFace dataset, with the following
            keys:
            - name_or_path: The name or path to a local dataset or a HuggingFace dataset.
            - primary_key: The primary key of the dataset. Default is 'image_key'.
            - column_mapping: The mapping of the dataset columns.
            - match_columns: Indicates whether to match the dataset columns. If True, columns not in the column_mapping will be removed.
            - fp_key: The key for the file path in the dataset. Default is 'image_path'. Applicable only for local datasets.
            - exts: The extensions of the images. Default is common image extensions ('jpg', 'jpeg', 'png', 'webp', 'jfif'). Applicable only for local datasets.
            - recur: Specifies whether to recursively search for images in the directory. Default is True. Applicable only for local datasets.
            - tbname: The table name of the SQLite3 dataset. Default is None, which means the dataset is assumed to be in a single table format. Applicable only for SQLite3 datasets.
            - dataset_type: Literal['local', 'huggingface', 'coco']: The type of the dataset. Default is None.
        - Dataset: An instance of the Dataset class.
    @param merge_mode: The mode to merge multiple datasets. Default is 'union'.
    @param dataset_cls: An optional class to use for the dataset. If not provided, a default dataset class will be used.
    @param **default_kwargs: Additional keyword arguments to pass to the dataset constructor.

    @returns: An instance of the Dataset class.
    """
    default_kwargs = {**DEFAULT_KWARGS, **default_kwargs}
    verbose = default_kwargs.get('verbose', True)
    source = parse_source_input(source)
    if not source:
        from .dict_dataset import DictDataset
        return (dataset_cls or DictDataset).from_dict({})
    datasets = []
    for i, src in enumerate(source):
        with logger.timer(f"Load dataset {i + 1}/{len(source)}") if verbose else contextlib.nullcontext():
            if isinstance(src, Dataset):
                dataset = src
                if not hasattr(dataset, 'priority'):
                    dataset.priority = i
                verbose_local = default_kwargs.get('verbose', False)

            else:
                src = dict(src)
                name_or_path = src.pop('name_or_path')
                dataset_type = src.pop('dataset_type', default_kwargs.get('dataset_type', None))
                # logger.debug(f"dataset type: {dataset_type}", disable=not verbose_local)
                primary_key = src.pop('primary_key', default_kwargs.get('primary_key'))
                if dataset_type == 'coco':
                    dataset = load_coco_dataset(
                        name_or_path,
                        primary_key=primary_key,
                        split=src.pop('split', default_kwargs.get('split')),
                        annotations=src.pop('annotations', default_kwargs.get('annotations')),
                        column_mapping=src.pop('column_mapping', default_kwargs.get('column_mapping')),
                        remove_columns=src.pop('remove_columns', default_kwargs.get('remove_columns')),
                        fp_key=src.pop('fp_key', default_kwargs.get('fp_key')),
                        read_attrs=src.pop('read_attrs', default_kwargs.get('read_attrs')),
                        verbose=src.pop('verbose', verbose),
                        **src,
                    )
                elif dataset_type == 'index_kits':
                    dataset = load_index_kits_dataset(
                        name_or_path,
                        additional_metadata_column=src.pop('additional_metadata_column', default_kwargs.get('additional_metadata_column')),
                        primary_key=primary_key,
                        verbose=src.pop('verbose', verbose),
                        **src,
                    )
                elif dataset_type == 'local' or os.path.exists(name_or_path) or os.path.splitext(name_or_path)[1] in ['.csv', '.json', '.sqlite3', '.db']:
                    dataset = load_single_dataset(
                        name_or_path,
                        primary_key=primary_key,
                        column_mapping=src.pop('column_mapping', default_kwargs.get('column_mapping')),
                        remove_columns=src.pop('remove_columns', default_kwargs.get('remove_columns')),

                        fp_key=src.pop('fp_key', default_kwargs.get('fp_key')),
                        recur=src.pop('recur', default_kwargs.get('recur')),
                        exts=src.pop('exts', default_kwargs.get('exts')),
                        tbname=src.pop('tbname', default_kwargs.get('tbname') if not os.path.exists(name_or_path) else None),
                        read_attrs=src.pop('read_attrs', default_kwargs.get('read_attrs')),
                        dataset_cls=None,
                        verbose=src.pop('verbose', verbose),
                        **src,
                    )
                else:
                    dataset = load_huggingface_dataset(
                        name_or_path=name_or_path,
                        primary_key=primary_key,
                        column_mapping=src.pop('column_mapping', default_kwargs.get('column_mapping')),
                        remove_columns=src.pop('remove_columns', default_kwargs.get('remove_columns')),

                        cache_dir=src.pop('cache_dir', default_kwargs.get('cache_dir')),
                        token=src.pop('token', default_kwargs.get('token')),
                        split=src.pop('split', default_kwargs.get('split')),
                        max_retries=src.pop('max_retries', default_kwargs.get('max_retries')),
                        dataset_cls=None,
                        verbose=src.pop('verbose', verbose),
                        **src,
                    )
                if (mapping := src.pop('mapping', None)) is not None:
                    dataset = mapping(dataset)
                dataset.priority = src.pop('priority', i)
                verbose_local = src.pop('verbose', False)

            datasets.append(dataset)
            logger.info(f"[{i}/{len(source)}] {dataset.name}:", disable=not verbose_local)
            logger.info(dataset, no_prefix=True, disable=not verbose_local)

    if merge_mode != 'chain':
        datasets.sort(key=lambda x: x.priority, reverse=True)
        dataset = accumulate_datasets(datasets, dataset_cls=dataset_cls, mode=merge_mode, verbose=verbose)
        if (mapping := default_kwargs.get('mapping', None)) is not None:
            dataset = mapping(dataset)
        dataset.register_to_config(**{k: v for k, v in default_kwargs.items() if k not in dataset.config})
    else:
        dataset = datasets
    return dataset


def load_single_dataset(
    name_or_path: str,
    primary_key: str = 'image_key',
    column_mapping: Dict[str, str] = None,
    remove_columns: List[str] = None,
    fp_key: str = 'image_path',
    exts: List[str] = IMAGE_EXTS,
    recur: bool = False,
    tbname: str = None,
    read_attrs: bool = False,
    verbose: bool = False,
    dataset_cls: type = None,
    **kwargs: Dict[str, Any],
):
    logger.info(f"Creating dataset object from '{logging.yellow(name_or_path)}'...", disable=not verbose)
    localset = AutoDataset(
        name_or_path,
        dataset_cls=dataset_cls,
        fp_key=fp_key,
        exts=exts,
        primary_key=primary_key,
        recur=recur,
        tbname=tbname,
        verbose=verbose,
        **kwargs,
    )
    if column_mapping:
        logger.info(f"Renaming columns: {', '.join(f'{logging.blue(k)} -> {logging.yellow(v)}' for k, v in column_mapping.items())}...", disable=not verbose)
        localset = localset.rename_columns(column_mapping, tqdm_disable=True)
    if remove_columns:
        logger.info(f"Removing columns: {', '.join(logging.yellow(remove_columns))}...", disable=not verbose)
        localset = localset.remove_columns(remove_columns, tqdm_disable=True)
    if hasattr(localset, 'primary_key'):
        primary_key = localset.primary_key
    if primary_key not in localset.headers:
        logger.warning(f"Primary key '{logging.yellow(primary_key)}' not found in the dataset. Patching the primary key...")
        localset = patch_key(localset, primary_key)
    if read_attrs:
        logger.info(f"Reading additional attributes...", disable=not verbose)
        if isinstance(localset, SQLite3Dataset):
            readset = localset.subset('caption', 'is NULL')
            readset.with_map(mapping.attr_reader, tqdm_disable=not verbose)
            localset.update(readset)
        else:
            localset.apply_map(mapping.attr_reader, condition=lambda img_md: img_md.get('caption') is None, tqdm_disable=not verbose)
    return localset


def load_huggingface_dataset(
    name_or_path: str,
    primary_key: str = 'image_key',
    column_mapping: Dict[str, str] = None,
    remove_columns: List[str] = None,
    cache_dir: str = None,
    token: str = None,
    split: str = 'train',
    max_retries: int = None,
    verbose: bool = False,
    dataset_cls: type = None,
    **kwargs: Dict[str, Any],
) -> Dataset:
    r"""
    Load dataset from HuggingFace and convert it to `dataset_cls`.
    """
    import datasets
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError("Please install huggingface-hub by `pip install huggingface-hub` to load dataset from HuggingFace.")
    if dataset_cls is None:
        from .dict_dataset import DictDataset
        dataset_cls = DictDataset
    if isinstance(column_mapping, (list, tuple)):
        column_mapping = {k: k for k in column_mapping}
    retries = 0
    while True:
        try:
            hfset: datasets.Dataset = datasets.load_dataset(
                name_or_path,
                cache_dir=cache_dir,
                split=split,
                token=token,
                **kwargs,
            )
            break
        except (huggingface_hub.utils.HfHubHTTPError, ConnectionError, requests.exceptions.HTTPError, requests.exceptions.ReadTimeout):
            logger.info(logging.yellow(f"Connection error when downloading dataset `{name_or_path}` from HuggingFace. Retrying..."))
            if max_retries is not None and retries >= max_retries:
                raise
            retries += 1
            pass

    if remove_columns:
        hfset = hfset.remove_columns([k for k in hfset.column_names if k in remove_columns])

    column_mapping = column_mapping or {}
    if isinstance(column_mapping, (list, tuple)):
        column_mapping = {k: k for k in column_mapping}
    if '__key__' in hfset.column_names:
        column_mapping['__key__'] = primary_key
    if column_mapping:
        hfset = hfset.rename_columns({k: v for k, v in column_mapping.items() if k != v and k in hfset.column_names})

    dic = {}
    if primary_key not in hfset.column_names:
        for index in range(len(hfset)):
            img_key = str(index)
            dic[img_key] = HuggingFaceData(hfset, index=index, **{primary_key: img_key})
    else:
        for index, img_key in enumerate(hfset[primary_key]):
            dic[img_key] = HuggingFaceData(hfset, index=index, **{primary_key: img_key})
    return dataset_cls.from_dict(dic, verbose=verbose)


def load_index_kits_dataset(
    name_or_path: str,
    additional_metadata_column: str = None,
    primary_key: str = 'image_key',
    verbose: bool = False,
    **kwargs: Dict[str, Any],
) -> Dataset:
    r"""
    Load dataset from Index Kits and convert it to `Dataset`.
    """
    from .index_kits_dataset import IndexKitsDataset
    return IndexKitsDataset.from_disk(
        name_or_path,
        additional_metadata_column=additional_metadata_column,
        primary_key=primary_key,
        verbose=verbose,
        **kwargs
    )


def load_coco_dataset(
    name_or_path: str,
    split: Literal['train', 'val', 'test'] = 'train',
    annotations: List[Literal['instances', 'captions', 'person_keypoints']] = ['instances', 'captions'],
    primary_key: str = 'image_key',
    column_mapping: Dict[str, str] = None,
    remove_columns: List[str] = None,
    fp_key: str = 'image_path',
    read_attrs: bool = False,
    verbose: bool = False,
    dataset_cls: type = None,
    **kwargs: Dict[str, Any],
) -> Dataset:
    r"""
    Load the COCO dataset and convert it into a `Dataset` object.

    @param name_or_path: The root directory of the COCO dataset.
    @param split: The dataset split to load ('train', 'val', or 'test'). Default is 'train'.
    @param primary_key: The primary key of the dataset. Default is 'image_key'.
    @param column_mapping: A dictionary mapping original column names to new names.
    @param remove_columns: A list of column names to remove from the dataset.
    @param fp_key: The key for the file path in the dataset. Default is 'image_path'.
    @param exts: The extensions of the images. Default is common image extensions.
    @param read_attrs: Whether to read additional image attributes. Default is False.
    @param verbose: Whether to print verbose output. Default is False.
    @param dataset_cls: An optional class to use for the dataset. If not provided, `DictDataset` will be used.
    @param **kwargs: Additional keyword arguments.

    @returns: An instance of the `Dataset` class containing the COCO dataset.
    """
    import os
    from pycocotools.coco import COCO

    if dataset_cls is None:
        from .dict_dataset import DictDataset
        dataset_cls = DictDataset

    data_dir = os.path.join(name_or_path, f"{split}")
    cocos = {}

    for ann in annotations:
        ann_file = os.path.join(name_or_path, f"annotations/{ann}_{split}.json")
        cocos[ann] = COCO(ann_file)
    img_ids = cocos['instances'].getImgIds()
    cat_ids = cocos['instances'].getCatIds()

    data_dict = {}

    for img_id in logger.tqdm(img_ids, desc='loading COCO dataset', disable=not verbose):
        img_info = cocos['instances'].loadImgs(img_id)[0]
        image_path = os.path.join(data_dir, img_info['file_name'])

        # annotations_infos = {}
        # for ann in annotations:
        #     ann_ids = cocos[ann].getAnnIds(imgIds=img_id, catIds=cat_ids)
        #     anns = cocos[ann].loadAnns(ann_ids)
        #     annotations_infos[ann] = anns

        img_key = str(img_id)

        data = {
            primary_key: img_key,
            fp_key: image_path,
            'img_id': img_id,
            'cat_ids': cat_ids,
            'cocos': cocos,
        }
        data.update(img_info)  # Include additional image info

        data_dict[img_key] = data

    dataset = dataset_cls.from_dict(data_dict, verbose=verbose)

    if column_mapping:
        dataset = dataset.rename_columns(column_mapping, tqdm_disable=not verbose)
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns, tqdm_disable=not verbose)
    if primary_key not in dataset.headers:
        dataset = patch_key(dataset, primary_key)

    if read_attrs:
        dataset.apply_map(mapping.attr_reader, condition=lambda img_md: img_md.get('caption') is None, tqdm_disable=not verbose)

    return dataset


def parse_source_input(source: Union[List, Tuple, Dict, str, Path, None]) -> List[Dict[str, Any]]:
    if source is None:
        return []
    if not isinstance(source, (list, tuple)):
        source = [source]
    elif len(source) == 1 and isinstance(source[0], (list, tuple)):
        source = source[0]
    source = [
        dict(name_or_path=src) if isinstance(src, (str, Path))
        else src
        for src in source
    ]
    return source


def check_parsed_source(source: List[Dict[str, Any]]) -> None:
    for i, src in enumerate(source):
        if isinstance(src, Dataset):
            continue
        if 'name_or_path' not in src:
            raise ValueError(f"source[{i}] must contain 'name_or_path'")
        for key, value in src.items():
            if key != 'name_or_path' and key not in DEFAULT_KWARGS:
                raise ValueError(f"Invalid keyword argument: {key}={value}")


def patch_key(dataset, primary_key) -> Dataset:
    for key, value in dataset.logger.tqdm(dataset.items(), desc='patch primary_key'):
        value[primary_key] = key
    if 'header' in dataset.__dict__ and primary_key not in dataset.header:
        dataset.header.append(primary_key)
    return dataset


# def accumulate_datasets(datasets: List[Dataset], mode: Literal['union', 'intersection', 'update'] = 'union', verbose=True) -> Dataset:
#     r"""
#     Accumulate multiple datasets into a single dataset.

#     The overlapping order is determined by the order of the input datasets. The former dataset has a higher priority.
#     """
#     if not isinstance(datasets, Iterable) or isinstance(datasets, str):
#         raise TypeError(f"datasets must be an iterable of Dataset, not {type(datasets)}")
#     for i, ds in enumerate(datasets):
#         if not isinstance(ds, Dataset):
#             raise TypeError(f"datasets[{i}] must be an instance of Dataset, not {type(ds)}")

#     # If datasets is empty, return an empty DictDataset
#     if not datasets:
#         return DictDataset.from_dict({})
#     # If there is only one dataset, return itself
#     elif len(datasets) == 1:
#         return datasets[0]
#     # If types of some datasets are inconsistent, use DictDataset by default
#     elif not all(ds.__class__ == datasets[0].__class__ for ds in datasets):
#         logger.warning(f"because some types of datasets are inconsistent when accumulating, use {DictDataset.__name__} by default")
#         dataset_cls = DictDataset
#     # Otherwise, use the type of all these datasets
#     else:
#         dataset_cls = datasets[0].__class__

#     # Merge the configurations of all datasets
#     pivot_config = {}
#     for ds in datasets:
#         pivot_config.update(ds.config)
#     pivot_config.pop('name', 'FastDataset')
#     pivot_config['verbose'] = verbose

#     pivotset = dataset_cls.from_dataset(datasets[0], **pivot_config)
#     if 'header' in pivotset.__dict__:
#         del pivotset.__dict__['header']
#     if mode == 'union':
#         img_keys = set()
#         for ds in datasets:
#             img_keys.update(ds.keys())
#             if any(img_key is None for img_key in img_keys):
#                 logger.error(f"Dataset with None image key: {ds}")
#     elif mode == 'intersection':
#         img_keys = set(datasets[0].keys())
#         for ds in datasets[1:]:
#             img_keys.intersection_update(ds.keys())
#             if any(img_key is None for img_key in img_keys):
#                 logger.error(f"Dataset with None image key: {ds}")
#     elif mode == 'update':
#         pass
#     else:
#         raise ValueError(f"Invalid merge mode: {mode}")

#     for i, ds in logger.tqdm(enumerate(datasets[1:]), desc='accumulate datasets', position=1, disable=not verbose):
#         if mode == 'update':
#             pivotset.update(ds)
#         else:  # mode in ['union', 'intersection']
#             for img_key in logger.tqdm(img_keys, desc=f'accumulate {i + 1}/{len(datasets[1:])}-th dataset', position=2, disable=not verbose):
#                 if (new_img_md := ds.get(img_key)) is not None:
#                     # High level data prioritize low level data when they conflict
#                     if (old_img_md := pivotset.get(img_key)) is not None:
#                         old_img_md.update(new_img_md)
#                         if issubclass(new_img_md.__class__, old_img_md.__class__):
#                             new_img_md.update(old_img_md)
#                             pivotset[img_key] = new_img_md
#                     else:
#                         pivotset[img_key] = new_img_md

#     if mode == 'intersection':
#         for img_key in logger.tqdm(list(pivotset.keys()), desc='remove data', position=2, disable=not verbose):
#             if img_key not in img_keys:
#                 del pivotset[img_key]

#     return pivotset


def accumulate_datasets(datasets: List[Dataset], mode: Literal['union', 'intersection', 'update'] = 'union', dataset_cls=None, verbose=True) -> Dataset:
    r"""
    Accumulate multiple datasets into a single dataset.

    The overlapping order is determined by the order of the input datasets. The former dataset has a higher priority.
    """
    if not isinstance(datasets, Iterable) or isinstance(datasets, str):
        raise TypeError(f"datasets must be an iterable of Dataset, not {type(datasets)}")
    for i, ds in enumerate(datasets):
        if not isinstance(ds, Dataset):
            raise TypeError(f"datasets[{i}] must be an instance of Dataset, not {type(ds)}")

    # If datasets is empty, return an empty DictDataset
    if not datasets:
        return DictDataset.from_dict({})
    # If there is only one dataset, return itself
    elif len(datasets) == 1:
        return datasets[0]
    elif dataset_cls is not None:
        pass
    # If types of some datasets are inconsistent, use DictDataset by default
    elif not all(ds.__class__ == datasets[0].__class__ for ds in datasets):
        logger.warning(f"because some types of datasets are inconsistent when accumulating, use {DictDataset.__name__} by default")
        dataset_cls = DictDataset
    # Otherwise, use the type of all these datasets
    else:
        dataset_cls = datasets[0].__class__

    for ds in datasets:
        if not hasattr(ds, 'priority'):
            ds.priority = 0

    logger.info(f"Accumulating datasets:", disable=not verbose)
    logger.info(f"  Total number of datasets: {len(datasets)}", disable=not verbose, no_prefix=True)
    logger.info(f"  Merge mode: {mode}", disable=not verbose, no_prefix=True)

    if mode == 'intersection':
        # Find the dataset with the smallest size as the pivotset
        pivot_index = 0
        pivotset = datasets[pivot_index]
        for i, ds in enumerate(datasets):
            if len(ds) < len(pivotset) or (len(ds) == len(pivotset) and (ds.priority < pivotset.priority or len(ds.headers) < len(pivotset.headers))):
                pivotset = ds
                pivot_index = i
        pivot_priority = pivotset.priority
        pivotset = dataset_cls.from_dataset(pivotset) if pivotset.__class__ != dataset_cls else pivotset
        pivotset.priority = pivot_priority
        datasets.pop(pivot_index)

        # Accumulate the datasets
        for ds in logger.tqdm(datasets, desc='intersection datasets', position=0, disable=not verbose, leave=False, total=len(datasets)):
            for img_key in logger.tqdm(list(pivotset.keys()), desc='intersection data', position=1, disable=not verbose, leave=False, total=len(pivotset)):
                if img_key not in ds:
                    del pivotset[img_key]
                else:
                    old_img_md = pivotset[img_key]
                    new_img_md = ds[img_key]
                    if issubclass(new_img_md.__class__, old_img_md.__class__):  # if new is a same or sub class of old
                        if pivotset.priority <= ds.priority:  # if new has higher priority
                            old_img_md.update(new_img_md)  # overwrite old with new
                            pivotset[img_key] = new_img_md.__class__(old_img_md) if old_img_md.__class__ != new_img_md.__class__ else old_img_md
                        else:  # if old has higher priority
                            new_img_md.update(old_img_md)  # overwrite new with old
                            pivotset[img_key] = new_img_md  # keep new class
                    else:  # if old is a sub class of new
                        if pivotset.priority > ds.priority:  # if old has higher priority
                            new_img_md.update(old_img_md)  # overwrite new with old
                            pivotset[img_key] = old_img_md.__class__(new_img_md)  # init old class with new data
                        else:  # if new has higher priority
                            old_img_md.update(new_img_md)  # overwrite old with new
                            pivotset[img_key] = old_img_md  # keep old class

    elif mode == 'union':
        pivot_index = 0
        pivotset = datasets[pivot_index]
        for i, ds in enumerate(datasets):
            if len(ds) > len(pivotset) or (len(ds) == len(pivotset) and (ds.priority > pivotset.priority or len(ds.headers) > len(pivotset.headers))):
                pivotset = ds
                pivot_index = i
        pivot_priority = pivotset.priority
        pivotset = dataset_cls.from_dataset(pivotset) if pivotset.__class__ != dataset_cls else pivotset
        pivotset.priority = pivot_priority
        datasets.pop(pivot_index)

        for ds in logger.tqdm(datasets, desc='union datasets', position=0, disable=not verbose, leave=False, total=len(datasets)):
            for img_key in logger.tqdm(ds.keys(), desc='union data', position=1, disable=not verbose, leave=False, total=len(ds)):
                if img_key not in pivotset:
                    pivotset[img_key] = ds[img_key]
                else:
                    old_img_md = pivotset[img_key]
                    new_img_md = ds[img_key]
                    if issubclass(new_img_md.__class__, old_img_md.__class__):  # if new is a same or sub class of old
                        if pivotset.priority <= ds.priority:  # if new has higher priority
                            old_img_md.update(new_img_md)  # overwrite old with new
                            pivotset[img_key] = new_img_md.__class__(old_img_md) if old_img_md.__class__ != new_img_md.__class__ else old_img_md  # init new class with old data
                        else:  # if old has higher priority
                            new_img_md.update(old_img_md)  # overwrite new with old
                            pivotset[img_key] = new_img_md  # keep new class
                    else:  # if old is a sub class of new
                        if pivotset.priority > ds.priority:  # if old has higher priority
                            new_img_md.update(old_img_md)  # overwrite new with old
                            pivotset[img_key] = old_img_md.__class__(new_img_md)  # init old class with new data
                        else:  # if new has higher priority
                            old_img_md.update(new_img_md)  # overwrite old with new
                            pivotset[img_key] = old_img_md  # keep old class

    elif mode == 'update':
        pivotset = dataset_cls.from_dataset(datasets[0])
        pivotset.priority = datasets[0].priority

        for ds in logger.tqdm(datasets[1:], desc='update datasets', position=0, disable=not verbose, leave=False):
            pivotset.update(ds, tqdm_desc='update data', tqdm_position=1, tqdm_disable=not verbose, tqdm_leave=False)

    logger.info(f"Accumulation done.", disable=not verbose)
    return pivotset
