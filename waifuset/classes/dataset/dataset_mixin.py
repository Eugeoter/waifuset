from abc import abstractmethod, ABC
from typing import Dict, Any
from ...const import StrPath


class ConfigMixin(object):
    DEFAULT_CONFIG = {}

    def __init__(self, *args, **kwargs):
        for key, val in self.DEFAULT_CONFIG.copy().items():
            self.config.setdefault(key, val)

    def __getattr__(self, name):
        if name == 'config':
            self.config = {}
            return self.config

    def register_to_config(self, **kwargs):
        self.config.update(kwargs)


class FromConfigMixin(ConfigMixin, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], *args, **kwargs):
        pass


class FromDiskMixin(ConfigMixin):
    config = {
        **ConfigMixin.DEFAULT_CONFIG,
        'fp': None,
    }

    @classmethod
    @abstractmethod
    def from_disk(cls, fp: StrPath, *args, **kwargs):
        pass


class ToDiskMixin(object):
    @abstractmethod
    def dump(self, fp: StrPath, *args, **kwargs) -> None:
        pass


class DiskIOMixin(FromDiskMixin, ToDiskMixin):
    @abstractmethod
    def commit(self, *args, **kwargs) -> None:
        pass
