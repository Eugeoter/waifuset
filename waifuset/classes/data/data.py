import pandas as pd
import copy


class Data(object):
    def dict(self, flatten=False):
        d = {}
        for k, v in self.items():
            if isinstance(v, Data):
                for kk, vv in v.dict(flatten=flatten).items():
                    d[f'{k}.{kk}'] = vv
            else:
                d[k] = v
        return d

    def df(self):
        return pd.DataFrame([self])

    def copy(self):
        return copy.deepcopy(self)
