from __future__ import annotations
import torch
from torch import Tensor
import warnings
from .linrgb import linRGB


class sRGB:
    def _from_linRGB(self, color: Tensor) -> Tensor:
        thres = 0.0031308
        a = 0.055

        if color.min() < 0 or color.max() > 1:
            warnings.warn(
                "When converting for linRGB to sRGB, values should be in "
                f"range [0, 1] not [{color.min()} {color.max()}]"
            )

        color_clipped = torch.clip(color, 0.0, 1.0)
        color_clipped_f = color_clipped.reshape(-1)

        for y in range(0, color_clipped.size, 4096):
            fragment = color_clipped_f[y : y + 4096]
            low = fragment <= thres

            fragment[low] *= 12.92
            fragment[~low] = (1 + a) * fragment[~low] ** (1 / 2.4) - a

        return color_clipped

    def _from_XYZ(self, color: Tensor) -> Tensor:
        assert src.__class__.__name__ == "XYZ"
        linrgb = linRGB(self._illuminant_xyz)
        color = linrgb._from_XYZ(src, color)
        return self._from_linRGB(linrgb, color)

    def _to_XYZ(self, color: Tensor) -> Tensor:
        color_linRGB = linRGB()._from_sRGB(color)
        return linRGB()._to_XYZ(color_linRGB)
