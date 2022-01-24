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
        "l2-proj": (lambda x: 1 / norm(x, p=2), lambda x: -np.mean(x) / norm(x, p=2))
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


class Rescaling(Transform):

    def min_max(ratings):
        """ ratings : ratings of 1 voter """
        maxi, mini = max(ratings), min(ratings)
        ratings = (ratings - mini) / (maxi - mini)
        return ratings

    NAME2FUNC = {'min-max': lambda x : Rescaling.min_max(x)}

    def __init__(self, name='min-max'):
        self.ratings = Rescaling.NAME2FUNC[name]

    def apply(self, ratings):
        return self.ratings(ratings)


def norm(x, p=2):
    out = sum(np.abs(x) ** p) ** (1/p)
    return out
