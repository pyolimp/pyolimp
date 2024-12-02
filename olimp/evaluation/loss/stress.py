from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor


class STRESS(Module):
    """
    Computes the Stress or 1 - Stress metric between two tensors.

    Args:
        invert (bool): If True, computes `1 - Stress`. Default is False.
    """

    def __init__(self, invert: bool = False):
        super(STRESS, self).__init__()  # type: ignore
        self.invert = invert

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Stress metric (or 1 - Stress if invert is True).

        Args:
            x (Tensor): First input tensor.
            y (Tensor): Second input tensor.

        Returns:
            Tensor: The computed metric value.
        """
        # Compute intermediate values
        x_1 = torch.sum(x**2)
        y_2 = torch.sum(y**2)
        x_y = torch.sum(x * y)

        # Compute stress
        stress_value = torch.sqrt(1 - (x_y**2) / (x_1 * y_2))

        # Return stress or 1 - stress based on invert flag
        return 1 - stress_value if self.invert else stress_value
