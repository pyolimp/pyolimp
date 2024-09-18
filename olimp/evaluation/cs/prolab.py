from __future__ import annotations
from torch import Tensor
import torch


class ProLab:
    q = torch.tensor([0.7554, 3.8666, 1.6739])

    Q = (
        torch.tensor(
            [
                [75.54, 617.72, 48.34],
                [486.66, -595.45, 194.94],
                [167.39, -22.27, -243.28],
            ]
        )
        / 100.0
    )

    Q_inv = torch.tensor(
        [
            [0.13706328211735358, 0.13706328211735355, 0.13706328211735358],
            [0.1387382031383206, -0.024315485429340655, 0.008083459429919239],
            [0.08160688511070953, 0.09653291949249931, -0.3174818967768846],
        ]
    )

    def __init__(self, illuminant_xyz: Tensor):
        self._illuminant_xyz = illuminant_xyz

    def _from_XYZ(self, color: Tensor) -> Tensor:
        color_ = color / self._illuminant_xyz
        return (
            (torch.tensordot(color_, self.Q, dims=1)).T
            / (torch.tensordot(color_, self.q, dims=1).T + 1.0)
        ).T

    def _to_XYZ(self, color: Tensor) -> Tensor:
        y2 = torch.tensordot(color, self.Q_inv, dims=1)
        xyz = y2.T / (1.0 - torch.tensordot(y2, self.q, dims=1)).T
        return xyz.T * self._illuminant_xyz
