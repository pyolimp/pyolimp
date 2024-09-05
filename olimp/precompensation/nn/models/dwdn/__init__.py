from __future__ import annotations
import torch
from torch import nn, Tensor

from .model import DWDN


class PrecompensationDWDN(DWDN):
    def __init__(self, n_levels: int = 1, scale: float = 0):
        super().__init__(n_levels=n_levels, scale=scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, k: Tensor):
        x = super().forward(x, k)[0]
        return self.sigmoid(x)

    @classmethod
    def from_path(cls, path: str, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location=torch.get_default_device())
        model.load_state_dict(state_dict)
        return model
