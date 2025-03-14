from __future__ import annotations
import torch
from torch import Tensor


XYZ_TO_LMS = torch.tensor(
    (
        (0.8189330101, 0.3618667424, -0.1288597137),
        (0.0329845436, 0.9293118715, 0.0361456387),
        (0.0482003018, 0.2643662691, 0.6338517070),
    ),
    dtype=torch.float32,
)

LMS_TO_LAB = torch.tensor(
    (
        (0.2104542553, 0.7936177850, -0.0040720468),
        (1.9779984951, -2.4285922050, 0.4505937099),
        (0.0259040371, 0.7827717662, -0.8086757660),
    ),
    dtype=torch.float32,
)

XYZ_TO_LMS_INV = torch.tensor(
    (
        (1.2270138511035211, -0.5577999806518223, 0.2812561489664678),
        (-0.04058017842328059, 1.11225686961683, -0.0716766786656012),
        (-0.07638128450570689, -0.42148197841801266, 1.5861632204407947),
    ),
    dtype=torch.float32,
)

LMS_TO_LAB_INV = torch.tensor(
    (
        (0.9999999984505199, 0.3963377921737679, 0.2158037580607588),
        (1.0000000088817607, -0.10556134232365635, -0.06385417477170591),
        (1.0000000546724108, -0.08948418209496575, -1.2914855378640917),
    ),
    dtype=torch.float32,
)


class Oklab:
    def from_XYZ(self, color: Tensor) -> Tensor:
        lms = torch.tensordot(
            XYZ_TO_LMS.to(device=color.device), color, dims=1
        )
        lms_cubic_root = torch.pow(lms.clip(min=0.0), 1 / 3)
        return torch.tensordot(
            LMS_TO_LAB.to(device=color.device), lms_cubic_root, dims=1
        )

    def to_XYZ(self, color: Tensor) -> Tensor:
        lab = torch.tensordot(
            LMS_TO_LAB_INV.to(device=color.device), color, dims=1
        )
        lab_cube = torch.pow(lab, 3.0)
        return torch.tensordot(
            XYZ_TO_LMS_INV.to(device=color.device), lab_cube, dims=1
        )
