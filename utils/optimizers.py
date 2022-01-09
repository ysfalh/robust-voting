""" Optimizers class """
from scipy.optimize import minimize
import numpy as np


class Optimizer:
    def __init__(self, tolerance=1e-6, max_iter=100):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def minimize(self, function, bounds):
        raise NotImplementedError


class ShelfOptimizer(Optimizer):
    def __init__(self, tolerance=1e-6, max_iter=100):
        super().__init__(tolerance=tolerance, max_iter=max_iter)

    def minimize(self, function, bounds):
        minimum_obj = minimize(function, np.array([0]*len(bounds)), bounds=bounds, tol=self.tolerance,
                              options={"maxiter": self.max_iter, "disp": False})

        return minimum_obj.x
