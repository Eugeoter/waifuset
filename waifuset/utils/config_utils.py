from typing import get_type_hints
from absl import app, flags
from ml_collections import ConfigDict, config_flags


class FromConfigMixin(object):
    @classmethod
    def from_config(cls, config, **kwargs):
        # for k, v in get_full_type_hints(cls).items():
        for k, v in config.items():
            kwargs.setdefault(k, config[k])
        return cls(config, **kwargs)

    def __init__(self, config, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(self, k, v)


def config(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def run(func):
    def wrapper(argv):
        config = flags.FLAGS.config
        func(config)

    config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
    flags.mark_flags_as_required(["config"])
    app.run(wrapper)


def get_full_type_hints(cls):
    vars = get_type_hints(cls)
    for base in reversed(cls.__mro__):
        vars.update({k: type(v) for k, v in base.__dict__.items() if not k.startswith('__') and not callable(v)})
    return vars


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)
