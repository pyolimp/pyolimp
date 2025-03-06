from __future__ import annotations
import torch
from torch import Tensor
import warnings
from .linrgb import linRGB


class sRGB:
    def from_linRGB(self, color: Tensor) -> Tensor:
        thres = 0.0031308
        a = 0.055

        # if color.min() < 0 or color.max() > 1:
        #     warnings.warn(
        #         "When converting for linRGB to sRGB, values should be in "
        #         f"range [0, 1] not [{color.min()} {color.max()}]"
        #     )

        color_clipped = torch.clip(color, 0.0, 1.0)
        color_clipped_f = color_clipped.reshape(-1)

        for y in range(0, color_clipped.numel(), 16384):
            fragment = color_clipped_f[y : y + 16384]
            low = fragment <= thres

            fragment[low] *= 12.92
            fragment[~low] = (1 + a) * fragment[~low] ** (1 / 2.4) - a

        return color_clipped

    def from_XYZ(self, color: Tensor) -> Tensor:
        linrgb = linRGB(None)
        color = linrgb.from_XYZ(color)
        return self.from_linRGB(color)

    def to_XYZ(self, color: Tensor) -> Tensor:
        color_linRGB = linRGB().from_sRGB(color)
        return linRGB().to_XYZ(color_linRGB)
