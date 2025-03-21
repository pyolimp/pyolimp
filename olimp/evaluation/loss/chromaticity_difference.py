from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab as ProLabCS

def srgb2lab_chromaticity(srgb: Tensor) -> Tensor:
    lab = CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))
    lab_chromaticity = lab[1:3, :, :]
    return lab_chromaticity

def srgb2prolab_chromaticity(srgb: Tensor) -> Tensor:
    prolab = ProLabCS(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))
    prolab[0, :, :][prolab[0, :, :] == 0] = 1.0
    prolab_chromaticity = prolab[1:3, :, :] / prolab[0, :, :]
    return prolab_chromaticity

def CD_map(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
) -> Tensor:
    if color_space == "lab":
        chromaticity1 = srgb2lab_chromaticity(img1)
        chromaticity2 = srgb2lab_chromaticity(img2)

    elif color_space == "prolab":
        chromaticity1 = srgb2prolab_chromaticity(img1)
        chromaticity2 = srgb2prolab_chromaticity(img2)

    chromaticity_diff = torch.linalg.norm(chromaticity1 - chromaticity2, dim = 0)
    
    return chromaticity_diff

class CDBase(Module):
    _color_space: Literal["lab", "prolab"]

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        img1: Tensor,
        img2: Tensor,
    ) -> Tensor:
        assert img1.ndim == 4, img1.shape
        assert img2.ndim == 4, img2.shape

        assert img1.shape[1] == 3, img1.shape
        assert img2.shape[1] == 3, img2.shape

        cd_maps = torch.empty((img1.shape[0]))
        for idx in range(img1.shape[0]):
            cd_maps[idx] = torch.mean(
                CD_map(
                    img1[idx],
                    img2[idx],
                    self._color_space,
                )
            )
        return cd_maps


class Lab(CDBase):
    _color_space = "lab"


class ProLab(CDBase):
    _color_space = "prolab"
