import numpy as np


class Transform:
    def apply(self, param):
        raise NotImplementedError


class AffineTransform(Transform):
    NAME2TUPLE = {"identity": (lambda x: 1, lambda x: 0),
                  "standardization": (lambda x: 1/np.std(x), lambda x: -np.mean(x)/np.std(x)),
                  "min-max": (lambda x: 1/(max(x)-min(x)), lambda x: -min(x)/(max(x)-min(x))),
                  "mean-max": (lambda x: 1/(max(x)-min(x)), lambda x: -np.mean(x)/(max(x)-min(x)))
                  }

    def __init__(self, name="identity"):
        self.slope, self.offset = AffineTransform.NAME2TUPLE[name]

    def apply(self, param):
        out = self.slope(param) * param + self.offset(param)
        return out

    def sparse_apply(self, param, mask):
        slope = self.slope([x for i, x in enumerate(param) if mask[i] != 0])
        offset = self.offset([x for i, x in enumerate(param) if mask[i] != 0])
        out = slope * param + offset
        return out
