from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Literal, TypeAlias
import torchvision

from .model import DWDN

Input: TypeAlias = tuple[Tensor, Tensor]

class PrecompensationDWDN(DWDN):
    def __init__(self, n_levels: int = 2, scale: float = 0.5):
        super().__init__(n_levels=n_levels, scale=scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Input) -> Tensor:
        image, psf, = inputs
        image = super().forward(image, psf)[0]
        return self.sigmoid(image)

    @classmethod
    def from_path(cls, path: str, **kwargs):
        model = cls(**kwargs)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def preprocess(self, image: Tensor, psf: Tensor) -> Input:
        psf = torch.fft.fftshift(psf)
        return image, psf
        
    def arguments(self, input: Tensor, psf: Tensor):
        return {}
