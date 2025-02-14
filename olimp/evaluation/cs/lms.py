from __future__ import annotations
import torch
from torch import Tensor, tensor


class LMS:
    """
    1. http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    2. Fairchild, M. (2005). Color appearance in image displays, p. 177
    """

    VON_KRIES_M = tensor(
        (
            (0.40024, 0.70760, -0.08081),
            (-0.2263, 1.16532, 0.04570),
            (0.0, 0.0, 0.9182200),
        ),
    )
    VON_KRIES_M_INV = tensor(
        (
            (1.8599364, -1.1293816, 0.2198974),
            (0.3611914, 0.6388125, -0.0000064),
            (0.0, 0.0, 1.0890636),
        ),
    )

    def from_XYZ(self, color: Tensor) -> Tensor:
        return torch.tensordot(
            self.VON_KRIES_M.to(device=color.device), color, dims=1
        )

    def to_XYZ(self, color: Tensor) -> Tensor:
        return torch.tensordot(
            self.VON_KRIES_M_INV.to(device=color.device), color, dims=1
        )
