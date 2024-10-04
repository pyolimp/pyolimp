from __future__ import annotations
import torch
from torch import Tensor
import warnings


LIN_RGB_MATRIX = torch.tensor(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    )
).T


class linRGB:
    def __init__(self, illuminant_xyz: Tensor | None = None):
        assert illuminant_xyz is None

    def from_XYZ(self, color: Tensor) -> Tensor:
        return color @ LIN_RGB_MATRIX.to(device=color.device)

    def from_sRGB(self, color: Tensor) -> Tensor:
        if color.min() < 0 or color.max() > 1:
            warnings.warn(
                f"sRGB range should be in [0, 1] not [{color.min()}, {color.max()}]"
            )

        color = torch.where(
            color > 0.04045,
            torch.pow((color + 0.055) / 1.055, 2.4),
            color / 12.92,
        )
        return color

    def to_XYZ(self, color: Tensor) -> Tensor:
        return color @ torch.linalg.inv(LIN_RGB_MATRIX.to(device=color.device))
