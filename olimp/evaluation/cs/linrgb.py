from __future__ import annotations
import torch
from torch import Tensor
import warnings


LIN_RGB_MATRIX = torch.tensor(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    ),
)

LIN_RGB_MATRIX_INV = torch.tensor(
    (
        (0.4124564, 0.3575761, 0.1804375),
        (0.2126729, 0.7151522, 0.0721750),
        (0.0193339, 0.1191920, 0.9503041),
    ),
)


class linRGB:
    def __init__(self, illuminant_xyz: Tensor | None = None):
        assert illuminant_xyz is None

    def from_XYZ(self, color: Tensor) -> Tensor:
        return torch.tensordot(
            LIN_RGB_MATRIX.to(device=color.device), color, 1
        )

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
        return torch.tensordot(
            LIN_RGB_MATRIX_INV.to(device=color.device), color, 1
        )
