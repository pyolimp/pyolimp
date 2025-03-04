from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab as ProLabCS


def srgb2prolab(srgb: Tensor) -> Tensor:
    return ProLabCS(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def srgb2lab(srgb: Tensor) -> Tensor:
    return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def CD_map(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
    lightness_weight: int = 0,
) -> Tensor:
    from math import sqrt

    if color_space == "lab":
        lab1 = srgb2lab(img1)
        lab2 = srgb2lab(img2)

    elif color_space == "prolab":
        lab1 = srgb2prolab(img1)
        lab1[0, :, :][lab1[0, :, :] == 0] = 1.0
        lab1[1, :, :] = lab1[1, :, :] / lab1[0, :, :]
        lab1[2, :, :] = lab1[2, :, :] / lab1[0, :, :]

        lab2 = srgb2prolab(img2)
        lab2[0, :, :][lab2[0, :, :] == 0] = 1.0
        lab2[1, :, :] = lab2[1, :, :] / lab2[0, :, :]
        lab2[2, :, :] = lab2[2, :, :] / lab2[0, :, :]

    diff = lab1 - lab2
    weights = torch.tensor((sqrt(lightness_weight), 1, 1))[:, None, None]
    weighted_diff = diff * weights
    chromatic_diff = torch.linalg.norm(weighted_diff, dim=0)
    return chromatic_diff


class CDBase(Module):
    _color_space: Literal["lab", "prolab"]

    def __init__(self, lightness_weight: int = 0) -> None:
        super().__init__()
        self._lightness_weight = lightness_weight

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
                    self._lightness_weight,
                )
            )
        return cd_maps


class Lab(CDBase):
    _color_space = "lab"


class ProLab(CDBase):
    _color_space = "prolab"
