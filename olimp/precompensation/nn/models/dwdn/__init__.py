from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Any, TypeAlias

from .model import DWDN
from ..download_path import download_path, PyOlimpHF

Inputs: TypeAlias = tuple[Tensor, Tensor]


class PrecompensationDWDN(DWDN):
    """
    .. image:: ../../../../_static/dwdn.svg
       :class: full-width
    """

    def __init__(self, n_levels: int = 2, scale: float = 1.0):
        super().__init__(n_levels=n_levels, scale=scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Inputs) -> Tensor:
        (
            image,
            psf,
        ) = inputs
        image = super().forward(image, psf)[0]
        return (self.sigmoid(image),)

    @classmethod
    def from_path(cls, path: PyOlimpHF, **kwargs: Any):
        model = cls(**kwargs)
        path = download_path(path)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def preprocess(self, image: Tensor, psf: Tensor) -> Inputs:
        psf = torch.fft.fftshift(psf)
        return image, psf

    def postprocess(self, tensors: tuple[Tensor]) -> tuple[Tensor]:
        return tensors

    def arguments(self, input: Tensor, psf: Tensor):
        return {}
