from typing import Literal

import torch
from torch import Tensor

from ._base import ReducibleLoss, Reduction
from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab as ProLabCS


def _srgb2lab_chromaticity(srgb: Tensor) -> Tensor:
    lab = CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))
    lab_chromaticity = lab[1:3, :, :]
    return lab_chromaticity


def _srgb2prolab_chromaticity(srgb: Tensor) -> Tensor:
    prolab = ProLabCS(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))
    prolab[0, :, :][prolab[0, :, :] == 0] = 1.0
    prolab_chromaticity = prolab[1:3, :, :] / prolab[0, :, :]
    return prolab_chromaticity


class ChromaticityDifference(ReducibleLoss):
    def __init__(
        self,
        color_space: Literal["lab", "prolab"],
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if color_space == "lab":
            self._chromaticity = _srgb2lab_chromaticity
        elif color_space == "prolab":
            self._chromaticity = _srgb2prolab_chromaticity
        else:
            raise ValueError(color_space)

    def _loss(
        self,
        img1: Tensor,
        img2: Tensor,
    ) -> Tensor:
        assert img1.shape[0] == 3, img1.shape
        assert img2.shape[0] == 3, img2.shape

        chromaticity1 = self._chromaticity(img1)
        chromaticity2 = self._chromaticity(img2)

        return torch.mean(
            torch.linalg.norm(chromaticity1 - chromaticity2, dim=0)
        )
