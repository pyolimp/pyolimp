"""
HQS Blocks to solve subproblems
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.fft as fft
from torch import Tensor

from typing import Literal


class W_solver(nn.Module):
    """Implementation of w-subproblem solver"""

    def __init__(self, norm_type: Literal["l2", "l1", "l0"]) -> None:
        """Class comstructor"""
        super(W_solver, self).__init__()

        self.norm_type = norm_type.lower()

    @staticmethod
    def prox_l0(x: Tensor, const_: Tensor) -> Tensor:
        """Proximal operator for L0 norm"""

        return torch.where(x**2 > 2 * const_, x, torch.zeros_like(x))

    @staticmethod
    def prox_l1(x: Tensor, const_: Tensor) -> Tensor:
        """Proximal operator for L1 norm"""

        return torch.max(
            torch.abs(x) - const_, torch.zeros_like(x)
        ) * torch.sign(x)

    @staticmethod
    def prox_l2(x: Tensor, const_: Tensor) -> Tensor:
        """Proximal operator for L2 norm"""

        return x / (2 * const_ + 1)

    def forward(self, z: Tensor, d_fft: Tensor, const_: Tensor) -> Tensor:
        """W-subproblem solver

        Parameters:
            z (Tensor):
                Is a solution of Z-subproblem on previos iteration.
                Shape is `(B, C, H, W)` where B is a batch size.

            d_fft (Tensor):
                FFT from list of D-filters.
                Shape is `(B, N, H, W)` where B is a batch size, N is a number of filters.

            const_ (Tensor):
                Coefficient of proximal operator.
                Alias to `theta/gamma`.
                Shape is `(B,)` where B is a batch size.
        """

        # convolution Di*z
        d_fft = d_fft.unsqueeze(1)
        z_fft = fft.fft2(z).unsqueeze(2)
        dz_fft = fft.ifft2(d_fft * z_fft).real

        if self.norm_type == "l0":
            return self.prox_l0(dz_fft, const_)
        elif self.norm_type == "l1":
            return self.prox_l1(dz_fft, const_)
        elif self.norm_type == "l2":
            return self.prox_l2(dz_fft, const_)
        else:
            raise ValueError(
                f"Unexpected norm type: {self.norm_type}. Expect one of l0, l1, l2."
            )


class Z_solver(nn.Module):
    """Implementation of z-subproblem solver"""

    def __init__(self):
        """Class comstructor"""
        super(Z_solver, self).__init__()

    def forward(
        self,
        p: Tensor,
        kt_fft: Tensor,
        kernel_fft: Tensor,
        w: Tensor,
        d_fft: Tensor,
        gamma: Tensor,
        beta: Tensor,
    ):
        """Solution of z-subproblem

        Parameters:

            p (Tensor):
                Solution of P-subproblem.
                Shape is `(B, C, H, W)` where B is a batch size.

            kt_fft (Tensor):
                FFT from producing the transposed blur kernel K to the source image t.
                Shape is `(B, C, H, W)` where B is a batch size.

            k_fft (Tensor):
                FFT from blur kernel K.
                Shape is `(B, 1, H, W)` where B is a batch size.

            w (Tensor):
                Solution of w-problem.
                Shape is `(B, C, N, H, W)` where B is a batch size, N is a number of filters.

            d_fft (Tensor):
                FFT from list of D-filters.
                Shape is `(B, N, H, W)` where B is a batch size, N is a number of filters.

            beta (Tensor):
               Coefficient related to p.
               Shape is `(B,)` where B is a batch size.

            gamma (Tensor):
               Coefficient related to w.
               Shape is `(B,)` where B is a batch size.
        """

        # fft of product DT*D
        d_fft_conj = torch.conj(d_fft)
        dd_fft = torch.sum(d_fft_conj * d_fft, dim=1, keepdim=True)

        # fft of product D*w
        d_fft_conj = d_fft_conj.unsqueeze(1)
        dw_fft = torch.sum(d_fft_conj * fft.fft2(w), dim=2)

        num = kt_fft + gamma * dw_fft + beta * fft.fft2(p)
        den = torch.conj(kernel_fft) * kernel_fft + gamma * dd_fft + beta

        ratio = torch.where(den != 0, num / den, 0)

        return fft.ifft2(ratio).real
