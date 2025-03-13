"""
https://en.wikipedia.org/wiki/Lab_color_space
"""

from __future__ import annotations
import torch
from torch import Tensor


def f(t: Tensor):
    delta = 6 / 29
    return torch.where(
        t > delta**3,
        1.16 * torch.pow(t, 1 / 3) - 0.16,
        0.01 * t / (delta / 2) ** 3,
    )


def finv(t: Tensor):
    delta = 6 / 29
    t_ = t / 1.16
    return torch.where(t_ > 0.02 / 0.29, (t_ + 4 / 29) ** 3, 3 * delta**2 * t_)


class CIELAB:
    A = torch.tensor(
        (
            (0.0, 1.0, 0.0),
            (125 / 29, -125 / 29, 0.0),
            (0.0, 50 / 29, -50 / 29),
        )
    )
    Ainv = torch.tensor(
        (
            (1.0, 29 / 125, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, -116 / 200),
        )
    )

    _illuminant_xyz: Tensor

    def __init__(self, illuminant_xyz: Tensor):
        assert illuminant_xyz is not None
        self._illuminant_xyz = illuminant_xyz

    def from_XYZ(self, color: Tensor):
        illuminant_xyz = self._illuminant_xyz.view(
            3, *((1,) * (color.dim() - 1))
        )
        return torch.tensordot(
            self.A.to(device=color.device),
            f(color / illuminant_xyz.to(device=color.device)),
            dims=1,
        )

    def to_XYZ(self, color: Tensor):
        illuminant_xyz = self._illuminant_xyz.view(
            3, *((1,) * (color.dim() - 1))
        )
        return (
            finv(
                torch.tensordot(
                    self.Ainv.to(device=color.device), color, dims=1
                )
            )
            * illuminant_xyz
        )
