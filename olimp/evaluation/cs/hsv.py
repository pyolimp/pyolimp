from __future__ import annotations
from .hls import HLS
import torch
from torch import Tensor


class HSV:
    """
    https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    def from_sRGB(self, color: Tensor) -> Tensor:
        h, l, s = HLS().from_sRGB(color)
        v = l + s * torch.minimum(l, 1.0 - l)
        s = 2 * (1.0 - torch.nan_to_num(torch.divide(l, v, out=l), 1.0, out=l))
        return torch.stack((h, s, v)).reshape(color.shape)

    def to_sRGB(self, color: Tensor) -> Tensor:
        h, s, v = color
        l = v * (1.0 - s / 2.0)
        minl = torch.minimum(l, 1.0 - l)
        s = torch.nan_to_num(torch.divide((v - l), minl, out=minl), out=minl)

        return HLS().to_sRGB(torch.stack((h, l, s)).reshape(color.shape))
