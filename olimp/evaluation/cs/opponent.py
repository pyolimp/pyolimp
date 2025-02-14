from __future__ import annotations
import torch
from torch import Tensor


class Opponent:
    """
    https://sid.onlinelibrary.wiley.com/doi/pdf/10.1889/1.1985127?casa_token=B5Gc5J-fFsgAAAAA:rgihZLGZbu2AnpZq6_Gxj78PG4fA0Eh4sPpW5A8Pkg4WJ-7J_rkyJhcxGUjMgS02XhFV8JGppZ22CkrZ
    """

    M = torch.tensor(
        (
            (0.279, -0.449, 0.086),
            (0.72, 0.29, -0.59),
            (-0.107, -0.077, 0.501),
        ),
        dtype=torch.float32,
    )

    Minv = torch.tensor(
        (
            (0.62655450425, 1.369855450124, 1.505650754905),
            (-1.867177597834, 0.93475582413, 1.421323771758),
            (-0.15315637341, 0.436229005232, 2.53602108024),
        ),
        dtype=torch.float32,
    )

    def from_XYZ(self, color: Tensor) -> Tensor:
        return torch.tensordot(self.M.T.to(device=color.device), color, dims=1)

    def to_XYZ(self, color: Tensor) -> Tensor:
        return torch.tensordot(
            self.Minv.to(device=color.device).T, color, dims=1
        )
