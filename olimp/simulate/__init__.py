from typing import TypeAlias
from torch import Tensor
from collections.abc import Callable

ApplyDistortion: TypeAlias = Callable[[Tensor], Tensor]
Distortion: TypeAlias = Callable[..., ApplyDistortion]
