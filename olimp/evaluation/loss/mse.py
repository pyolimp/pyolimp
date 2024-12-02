from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor


class MSE(Module):
    """
    Mean Squared Error (MSE) metric implemented as a PyTorch module.

    Args:
        invert (bool): If True, returns `1 - MSE` for compatibility with certain metrics.
    """

    def __init__(self, invert: bool = True) -> None:
        super(MSE, self).__init__()  # type: ignore
        self.invert = invert

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Mean Squared Error (MSE) between two tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: The computed MSE value. If invert is True, returns `1 - MSE`.
        """
        mse_value = torch.mean((x - y) ** 2)
        return 1 - mse_value if self.invert else mse_value
