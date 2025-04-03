from __future__ import annotations

import torch
from torch import Tensor
from typing import Literal

from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab
from ..cs.oklab import Oklab
from ._base import ReducibleLoss, Reduction, identity


def srgb2prolab(srgb: Tensor) -> Tensor:
    return ProLab(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def srgb2lab(srgb: Tensor) -> Tensor:
    return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def srgb2oklab(srgb: Tensor) -> Tensor:
    return Oklab().from_XYZ(sRGB().to_XYZ(srgb))


class RMSE(ReducibleLoss):
    """
    Root Mean Squared Error (RMSE) metric implemented as a PyTorch module.
    """

    def __init__(
        self,
        color_space: Literal["srgb", "lab", "prolab", "oklab"] = "srgb",
        reduction: Reduction = "mean",
    ):
        super().__init__(reduction=reduction)
        match color_space:
            case "lab":
                self.color_transform = srgb2lab
            case "prolab":
                self.color_transform = srgb2prolab
            case "oklab":
                self.color_transform = srgb2oklab
            case "srgb":
                self.color_transform = identity

    def _loss(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        Computes the Root Mean Squared Error (RMSE) between two tensors.

        Args:
            img1 (torch.Tensor): First input tensor.
            img2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: The computed RMSE value.
        """

        img1 = self.color_transform(img1)
        img2 = self.color_transform(img2)

        rmse_value = torch.linalg.norm(img1 - img2)
        return rmse_value
