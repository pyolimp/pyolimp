from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor
from .mse import MSE


class PSNR(Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric implemented as a PyTorch module.

    Args:
        mse_metric (Module): MSE metric class instance.
    """

    def __init__(
        self,
        mse_metric: MSE = MSE(),
    ) -> None:
        self.mse_metric = mse_metric

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Peak Signal-to-Noise Ratio (PSNR) between two tensors.

        Args:
            x (Tensor): First input tensor.
            y (Tensor): Second input tensor.

        Returns:
            Tensor: The computed PSNR value. Returns `inf` if MSE is zero.
        """

        mse_value = self.mse_metric(x, y)
        max_pixel = torch.max(x)

        if mse_value == 0:
            return torch.tensor(float("inf"), device=x.device)
        psnr_value = 10 * torch.log10((max_pixel**2) / mse_value)
        return psnr_value
