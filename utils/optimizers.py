""" Optimizers class """
from scipy.optimize import minimize
import numpy as np
import math


class Optimizer:
    def __init__(self, tolerance=1e-6, max_iter=100):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def minimize(self, function, bounds):
        raise NotImplementedError


class ShelfOptimizer(Optimizer):
    def __init__(self, tolerance=1e-9, max_iter=100):
        super().__init__(tolerance=tolerance, max_iter=max_iter)

    def minimize(self, function=None, bounds=(-1., 1.), **kwargs):
        minimum_obj = minimize(function, np.array([0] * len(bounds)), bounds=bounds, tol=self.tolerance,
                               options={"maxiter": self.max_iter, "disp": False})

        return minimum_obj.x


class Dichotomy(Optimizer):
    def __init__(self, tolerance=1e-9, max_iter=100):
        super().__init__(tolerance=tolerance, max_iter=max_iter)

    def minimize(self, derivative=None, bounds=((-1., 1.),), **kwargs):
        inf, sup = bounds[0]
        mid = (inf + sup) / 2
        n_iter = self.max_iter
        if bounds[0][1] != bounds[0][0]:
            n_iter = min(n_iter, int(math.log2(abs(bounds[0][1] - bounds[0][0]) / self.tolerance)))
        else:
            n_iter = 1
        for i in range(n_iter):
            if derivative(mid) > 0:
                sup = mid
            else:
                inf = mid
            mid = (inf + sup) / 2
        return mid


def derivate(x, weights, values, deltas=[], voting_resilience=1, default_val=0.):
    """ computes the derivative of QrMed """
    if not hasattr(deltas, '__len__'):  # if delta not custom for each user
        deltas = [deltas] * len(weights)
    deriv = voting_resilience * (x - default_val)
    for value, weight, delta in zip(values, weights, deltas):
        deriv += weight * sign(x - value) * (1 - math.exp(-abs(x - value) / delta)) if delta != 0 else weight * sign(x - value)
    return deriv


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return - 1
    return 0
