from __future__ import annotations
from typing import Literal
import torch
from torch import Tensor


Reduction = Literal["none", "mean", "sum"]


def identity(a: Tensor) -> Tensor:
    """
    https://en.wikipedia.org/wiki/Identity_function
    """
    return a


class ReducibleLoss(torch.nn.Module):
    """
    Base class for batch-unaware losses. For example, a loss function must
    compute `torch.max(x)`, but the maximum for the entire batch is likely not
    equal to the maximum of individual images.
    """

    def __init__(self, reduction: Reduction = "mean"):
        super().__init__()  # type: ignore
        match reduction:
            case "mean":
                self._reduction = torch.mean
            case "sum":
                self._reduction = torch.sum
            case "none":
                self._reduction = identity
            case _:
                raise ValueError(reduction)

    def _loss(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Call `_loss` for each batch
        """
        assert x.ndim == 4, x.shape
        assert y.ndim == 4, y.shape
        out = torch.empty(x.shape[0])
        for idx, (x, y) in enumerate(zip(x, y)):
            out[idx] = self._loss(x, y)
        return self._reduction(out)
