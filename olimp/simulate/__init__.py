from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class Distortion(ABC):
    @abstractmethod
    def __init__(self, *args: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, image: Tensor) -> Tensor:
        raise NotImplementedError
