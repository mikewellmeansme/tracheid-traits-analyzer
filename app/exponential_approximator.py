import numpy as np
import scipy.optimize as opt

from zhutils.approximators import Approximator
from typing import List, Callable


def exp_formula(x: np.array, y_min: float, y_as: float, a: float):
    return y_min + (y_as - y_min) * (1 - np.exp(-a*x))


class Exponential(Approximator):
    formula: Callable[[List], List]
    y_min: float
    __coeffs__: List

    def __init__(self, y_min: float) -> None:
        self.y_min = y_min

    def fit(self, x: List, y: List, **kwargs) -> None:

        p0 = [self.y_min, min(y), 0.1]
        bounds = ([self.y_min - .01 * self.y_min, min(y), 0], [self.y_min, max(y), 1])
        coeffs, _ = opt.curve_fit(exp_formula, x, y, p0=p0, maxfev=5000, bounds=bounds)
        self.__coeffs__ = coeffs

        self.formula = lambda l: coeffs[0] + (coeffs[1]-coeffs[0]) * (1 - np.exp(-coeffs[2]*np.array(l)))

    def predict(self, x: List) -> List:
        return self.formula(x)

    def get_equation(self, precision: int = 2) -> str:
        y_min, y_as, a = self.coeffs
        return f"$ {y_min:.{precision}f} + {y_as-y_min:.{precision}f}·e^{{−{a:.{precision}f}x}}$"

    @property
    def coeffs(self) -> List[float]:
        return self.__coeffs__
