from .dict_data import DictData


class HuggingFaceData(DictData):
    r"""
    A dictionary-like data object that supports huggingface datasets.
    """
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], dict):
            return args[0]
        return super().__new__(cls)

    def __init__(
        self,
        torch_dataset,
        index: int = None,
        **kwargs,
    ):
        self.host = torch_dataset
        self.index = int(index)
        super().__init__(**kwargs)

    def get(self, key, default=None):
        if key in self.host.column_names:
            return self.host[self.index].get(key, default)
        return super().get(key, default)

    def __getattr__(self, name):
        if 'host' in self.__dict__ and name in self.host.column_names:
            return self.host[self.index][name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        if key in self.host.column_names:
            return self.host[self.index][key]
        return super().__getitem__(key)
