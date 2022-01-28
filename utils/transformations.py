import numpy as np
import math


class Transform:
    def apply(self, param):
        raise NotImplementedError


class AffineTransform(Transform):
    NAME2TUPLE = {
        "identity": (lambda x: 1, lambda x: 0),
        "standardization": (lambda x: 1 / np.std(x), lambda x: -np.mean(x) / np.std(x)),
        "min-max": (lambda x: 1 / (max(x) - min(x)), lambda x: -min(x) / (max(x) - min(x))),
        "mean-max": (lambda x: 1 / (max(x) - min(x)), lambda x: -np.mean(x) / (max(x) - min(x))),
        "median-quartile": (lambda x: 1 / (np.quantile(x, .75) - np.quantile(x, .25)),
                            lambda x: - np.median(x) / (np.quantile(x, .75) - np.quantile(x, .25))),
        "l1-norm": (lambda x: 1 / norm(x, p=1), lambda x: 0),
        "l2-norm": (lambda x: 1 / norm(x, p=2), lambda x: 0),
        "l2-proj": (lambda x: 1 / norm(x - np.mean(x), p=2), lambda x: -np.mean(x) / norm(x - np.mean(x), p=2)),
        "adversarial-0.5": (lambda x: 1 / np.sqrt(np.abs(x - np.mean(x)).sum()),
                            lambda x: -np.mean(x) / np.sqrt(np.abs(x - np.mean(x)).sum())),
        "adversarial-0.8": (lambda x: 1 / np.abs(x - np.mean(x)).sum() ** 0.8,
                            lambda x: -np.mean(x) / np.abs(x - np.mean(x)).sum() ** 0.8)
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


def norm(x, p=2):
    out = sum(np.abs(x) ** p) ** (1 / p)
    return out
