"""
Loss Function
"""

from __future__ import annotations
import torch
import torch.fft as fft
from torch import Tensor

from typing import Literal

from .filters import grad_like, sobel_like


def align_bs(input: Tensor, new_bs: int) -> Tensor:
    """
    Align `input` batch size to `new_bs`
    Parameters:
        input (Tensor):
            Input tensor.
            Shape is `(B, C, H ,W)` where B is an old batch size.
        new_bs (int):
            New batch size.
    Returns:
        out (Tensor):
            Tensor with `new_bs` batch size.
    """
    single_t = input[0:1]
    return single_t.expand(new_bs, -1, -1, -1)


def fft_conv_5d(x: Tensor, y: Tensor) -> Tensor:
    """
    FFT base method for fast convolution with multiple channels and filters.
    Parameters:
        x (Tensor):
            Input image.
            Shape is `(B, C, H, W)` where `C` is a number of channels.
        y (Tensor):
            Array of filters.
            Shape is `(B, N, H, W)` where `N` is a number of filters.
    Returns:
        out (Tensor):
            Convolved image.
            Shape is `(B, C, N, H, W)`.
    """
    x_fft = fft.fft2(x).unsqueeze(2)
    y_fft = fft.fft2(y).unsqueeze(1)

    return torch.real(torch.fft.ifft2(x_fft * y_fft))


class PreCompensationLoss:
    """Loss function for image precompensation"""

    def __init__(
        self,
        theta: float = 0.0,
        filters: Tensor | Literal["sobel", "grad"] = "grad",
        regularization: Literal["none", "l2", "l1", "l0"] = "none",
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        """
        Class constructor
        Parameters:
            theta (float):
                Regularization weight.
                Default is `0.0`.
            filters (Tensor or str):
                If input is a Tensor, use this values to compute gradient.
                If input is one of `"sobel"`, `"grad"` use correspondance gradient filters.
                Default is `"grad"`.
            regularization (str):
                Regularization norm type: `"none"`, `"l2"`, `"l1"`, `"l0"`.
                Default is `"none"`.
            reduction (str):
                Batch reduction type: `"mean"` or `"sum"`.
                Default is `"mean"`.
        """
        self.theta = theta
        self.reg = regularization
        self.reduction = reduction
        self.filters = filters

    def _agr(self, x: Tensor) -> Tensor:
        """Apply agregation function to tensor values"""
        if self.reduction == "mean":
            return torch.mean(x) * x.size(1) * x.size(2)
        elif self.reduction == "sum":
            return torch.sum(x)
        else:
            raise ValueError(
                f"Unexpected reduction type: {self.reduction}. Expect one of mean, sum."
            )

    def _get_filtered(self, x: Tensor) -> Tensor:
        """Apply filters to source image"""

        if isinstance(self.filters, Tensor):
            self.filters = align_bs(self.filters, x.size(0))
            return fft_conv_5d(x, self.filters)
        if isinstance(self.filters, str):
            if self.filters == "grad":
                return fft_conv_5d(x, grad_like(x))
            elif self.filters == "sobel":
                return fft_conv_5d(x, sobel_like(x))
            else:
                raise ValueError(f"Invalid filters type: {self.filters}")

    def __call__(self, p: Tensor, t: Tensor, k: Tensor) -> Tensor:
        """
        Get optimization loss for image precompensation
        Parameters:
            p (Tensor):
                Precompensated image.
                Shape is `(B, C, H, W)`.
            t (Tensor):
                Source image.
                Shape is `(B, C, H, W)`.
            k (Tensor):
                Point Spread Function.
                Shape is `(B, 1, H, W)`.
        Results:
            out (Tensor):
                Calculated loss L(p, t, k).
        """
        t = t.unsqueeze(2)
        p_retinal = fft_conv_5d(p, k)
        p_filtered = self._get_filtered(p)

        if self.reg == "none":
            g = 0
        elif self.reg == "l0":
            g = self._agr((p_filtered != 0).float())
        elif self.reg == "l1":
            g = self._agr(p_filtered.abs())
        elif self.reg == "l2":
            g = self._agr(p_filtered**2)
        else:
            raise ValueError(
                f"Unexpected norm type: {self.reg}. Expect one of none, l1, l2."
            )

        return 0.5 * self._agr((p_retinal - t) ** 2) + self.theta * g
