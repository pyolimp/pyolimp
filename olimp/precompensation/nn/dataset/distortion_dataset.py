from __future__ import annotations
from typing import Callable, Generator

from torch import Tensor
from torch.utils.data import Dataset


class DistortionDataset(Dataset[Tensor]):
    def __init__(
        self, seed: int, size: int, generator: None | Callable[..., Tensor]
    ) -> None:
        self._seed = seed
        self._size = size
        self._generator = generator

    def __getitem__(self, index: int) -> Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        return self._size

    def apply(self) -> Callable[[Tensor], Generator[Tensor, None, None]]:
        raise NotImplementedError
