from zhutils.approximators import Approximator
from typing import List

# TODO: implement


class Exponential(Approximator):

    def fit(self, x: List, y: List, **kwargs) -> None:
        ...

    def predict(self, x) -> List:
        ...

    @property
    def equation(self) -> str:
        ...

    @property
    def coeffs(self) -> List[float]:
        ...
