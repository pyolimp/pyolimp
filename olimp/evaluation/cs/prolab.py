from __future__ import annotations
from torch import Tensor
import torch


class ProLab:
    q = torch.tensor((0.7554, 3.8666, 1.6739))

    Q = (
        torch.tensor(
            (
                (75.54, 486.66, 167.39),
                (617.72, -595.45, -22.27),
                (48.34, 194.94, -243.28),
            )
        )
        / 100.0
    )

    Q_inv = torch.tensor(
        (
            (0.13706328211735358, 0.1387382031383206, 0.08160688511070953),
            (0.13706328211735355, -0.024315485429340655, 0.09653291949249931),
            (0.13706328211735358, 0.008083459429919239, -0.3174818967768846),
        )
    )

    def __init__(self, illuminant_xyz: Tensor):
        self._illuminant_xyz = illuminant_xyz

    def from_XYZ(self, color: Tensor) -> Tensor:
        illuminant_xyz = self._illuminant_xyz.view(
            -1, *((1,) * (color.dim() - 1))
        ).to(device=color.device)
        color_ = color / illuminant_xyz
        return (
            torch.tensordot(self.Q.to(device=color.device), color_, dims=1)
        ) / (
            torch.tensordot(self.q.to(device=color.device), color_, dims=1)
            + 1.0
        )

    def to_XYZ(self, color: Tensor) -> Tensor:
        y2 = torch.tensordot(self.Q_inv, color, dims=1)
        xyz = y2 / (
            1.0 - torch.tensordot(self.q.to(device=color.device), y2, dims=1)
        )
        return xyz * self._illuminant_xyz.view(
            -1, *((1,) * (color.dim() - 1))
        ).to(device=color.device)
