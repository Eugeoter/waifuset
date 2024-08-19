from .dict_data import DictData


class HuggingFaceData(DictData):
    def __init__(
        self,
        torch_dataset,
        index: int,
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
        if name in self.host.column_names:
            return self.host[self.index][name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        if key in self.host.column_names:
            return self.host[self.index][key]
        return super().__getitem__(key)
