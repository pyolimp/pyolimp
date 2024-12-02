from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor


class Correlation(Module):
    """
    Computes the correlation coefficient between two tensors.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the correlation coefficient.

        Args:
            x (Tensor): First input tensor.
            y (Tensor): Second input tensor.

        Returns:
            Tensor: The computed correlation value.
        """
        # Small epsilon to avoid division by zero
        epsilon = 1e-8

        # Compute means and standard deviations
        x_mean, x_std = x.mean(), x.std() + epsilon
        y_mean, y_std = y.mean(), y.std() + epsilon

        # Compute covariance
        covar = ((x - x_mean) * (y - y_mean)).mean()

        # Compute correlation
        correl = covar / (x_std * y_std)

        return correl
