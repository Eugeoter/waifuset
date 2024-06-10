import datasets
from .dict_dataset import DictDataset
from typing import Optional, Union, Sequence, Mapping, Dict, overload
from datasets import Split, Features, DownloadConfig, DownloadMode, VerificationMode, Version


class HuggingFaceDataset(DictDataset):
    DEFAULT_CONFIG = {
        **DictDataset.DEFAULT_CONFIG,
        'repo_id': None,
        'split': None,
    }

    def __init__(self, source, **kwargs):
        DictDataset.__init__(self, source, **kwargs)

    @overload
    @classmethod
    def from_repo(
        cls,
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        ignore_verifications="deprecated",
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token="deprecated",
        task="deprecated",
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict] = None,
        trust_remote_code: bool = None,
        **config_kwargs,
    ):
        ...

    @classmethod
    def from_repo(cls, path, **kwargs):
        return cls(datasets.load_dataset(path, **kwargs), repo_id=path, **kwargs)
