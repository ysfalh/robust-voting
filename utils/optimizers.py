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
        for i in range(self.max_iter):
            if abs(derivative(mid)) <= self.tolerance:
                return mid
            if derivative(mid) > 0:
                sup = mid
            else:
                inf = mid
            mid = (inf + sup) / 2

        return mid


def derivate(x, weights, values, delta=1, voting_resilience=1):
    """ computes the derivative of QrMed """
    deriv = voting_resilience * x
    for value, weight in zip(values, weights):
        if x <= value:
            deriv += weight * (math.exp((x - value) / delta) - 1)
        else:
            deriv += weight * (1 - math.exp((value - x) / delta))
    return deriv
