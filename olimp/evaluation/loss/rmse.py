from __future__ import annotations

from torch.nn import Module
import torch
from torch import Tensor
from typing import Literal

from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab
from ..cs.oklab import Oklab


def srgb2prolab(srgb: Tensor) -> Tensor:
    return ProLab(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def srgb2lab(srgb: Tensor) -> Tensor:
    return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def srgb2oklab(srgb: Tensor) -> Tensor:
    return Oklab().from_XYZ(sRGB().to_XYZ(srgb))


class RMSE(Module):
    """
    Root Mean Squared Error (RMSE) metric implemented as a PyTorch module.
    """

    def __init__(
        self, color_space: Literal["srgb", "lab", "prolab", "oklab"] = "srgb"
    ):
        super().__init__()
        self.color_space = color_space

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        Computes the Root Mean Squared Error (RMSE) between two tensors.

        Args:
            img1 (torch.Tensor): First input tensor.
            img2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: The computed RMSE value.
        """

        if self.color_space == "lab":
            img1 = srgb2lab(img1)
            img2 = srgb2lab(img2)
        elif self.color_space == "prolab":
            img1 = srgb2prolab(img1)
            img2 = srgb2prolab(img2)
        elif self.color_space == "oklab":
            img1 = srgb2oklab(img1)
            img2 = srgb2oklab(img2)

        rmse_value = torch.linalg.norm(img1 - img2)
        return rmse_value
