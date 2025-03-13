from __future__ import annotations
import torch
from torch import Tensor


class HLS:
    """
    https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    def from_sRGB(self, color: Tensor) -> Tensor:
        maxc = color.max(dim=0).values
        minc = color.min(dim=0).values
        sumc = maxc + minc
        rangec = maxc - minc
        l = sumc * 0.5
        saturation = torch.divide(
            rangec, torch.where(l <= 0.5, sumc, 2.0 - sumc)
        )
        torch.nan_to_num(saturation, 0.0, out=saturation)
        c = torch.divide(maxc - color, rangec)
        torch.nan_to_num(c, 0.0, out=c)
        r = color[0]
        g = color[1]
        rc, gc, bc = c
        hue = torch.where(
            r == maxc,
            bc - gc,
            torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
        )
        hue = (hue / 6.0) % 1.0
        return torch.stack((hue, l, saturation)).reshape(color.shape)

    def to_sRGB(self, color: Tensor) -> Tensor:
        h, l, s = color
        m2 = torch.where(l <= 0.5, l * (1.0 + s), l + s - (l * s))
        m1 = 2.0 * l - m2

        def _v(m1: Tensor, m2: Tensor, hue: Tensor) -> Tensor:
            hue = hue % 1.0
            return torch.where(
                hue < 1 / 6,
                m1 + (m2 - m1) * hue * 6.0,
                torch.where(
                    hue < 0.5,
                    m2,
                    torch.where(
                        hue < 2 / 3, m1 + (m2 - m1) * (2 / 3 - hue) * 6.0, m1
                    ),
                ),
            )

        return torch.stack(
            (
                _v(m1, m2, h + 1 / 3),
                _v(m1, m2, h),
                _v(m1, m2, h - 1 / 3),
            )
        ).reshape(color.shape)
