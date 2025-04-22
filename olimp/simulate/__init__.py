from typing import TypeAlias, Any
from torch import Tensor
from collections.abc import Callable

ApplyDistortion: TypeAlias = Callable[[Tensor], Tensor]


class Distortion:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ApplyDistortion:
        raise NotImplementedError
