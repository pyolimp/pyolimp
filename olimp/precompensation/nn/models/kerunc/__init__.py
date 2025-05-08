from __future__ import annotations

import typing as tp

import torch
from torch import nn, Tensor
from typing import Any, TypeAlias

from .model import kernel_error_model
from ..download_path import download_path, PyOlimpHF

Inputs: TypeAlias = tuple[Tensor, Tensor]


class PrecompensationKERUNC(kernel_error_model):
    """
    .. image:: ../../../../_static/kerunc.svg
       :class: full-width
    """

    def __init__(self,
                 lmds: tp.List[float] = [0.005, 0.1],
                 layers: int = 1,
                 deep: int = 17,):
        super().__init__(lmds=lmds, layers=layers, deep=deep)
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
