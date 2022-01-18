import numpy as np


class Loss:
    def apply(self, param):
        raise NotImplementedError


class Huber(Loss):

    def apply(self, param, delta):
        return np.sqrt(param ** 2 + delta ** 2) - np.abs(delta)


class MrDist(Loss):
    PRIOR2FUNC = {
        "laplace": lambda x, delta: np.abs(x) + delta * (np.exp(-np.abs(x) / delta) - 1)
    }

    def __init__(self, prior="laplace", *args):
        self.prior = prior

    def apply(self, param, delta):
        return MrDist.PRIOR2FUNC[self.prior](param, delta)
