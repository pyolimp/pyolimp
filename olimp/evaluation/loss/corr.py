from __future__ import annotations

from torch.nn import Module
from torch import Tensor


class Correlation(Module):
    """
    Computes the correlation coefficient between two tensors.
    """

    def __init__(self, invert: bool = False):
        super().__init__()  # type: ignore
        self.invert = invert

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
        x_mean, x_std = x.mean(), x.std()
        y_mean, y_std = y.mean(), y.std()

        # Compute covariance
        covar = ((x - x_mean) * (y - y_mean)).mean()

        # Compute correlation
        correl = covar / (x_std * y_std + epsilon)

        if self.invert:
            correl = 1 - correl

        return correl
