from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor


class MSE(Module):
    """
    Mean Squared Error (MSE) metric implemented as a PyTorch module.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Mean Squared Error (MSE) between two tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: The computed MSE value.
        """
        mse_value = torch.mean((x - y) ** 2)
        return mse_value
