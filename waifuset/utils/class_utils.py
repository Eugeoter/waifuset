from typing import get_type_hints
from _thread import RLock

_NOT_FOUND = object()


class dict_cached_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")
        cache = instance
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.attrname] = val
                    except TypeError:
                        msg = (
                            f"The attribute on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.attrname!r} property."
                        )
                        raise TypeError(msg) from None
        return val


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


def get_full_type_hints(cls):
    vars = get_type_hints(cls)
    for base in reversed(cls.__mro__):
        vars.update({k: type(v) for k, v in base.__dict__.items() if not k.startswith('__') and not callable(v)})
    return vars
