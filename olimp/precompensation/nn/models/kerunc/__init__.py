from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Any, TypeAlias

from .model import kernel_error_model, KerUncArgs
from ..download_path import download_path, PyOlimpHF

Inputs: TypeAlias = tuple[Tensor, Tensor]


class PrecompensationKerUnc(kernel_error_model):
    """
    .. image:: ../../../../_static/kerunc.svg
       :class: full-width
    """

    def __init__(
        self,
        lmds: list[float] = [0, 0, 0, 0, 0],
        layers: int = 4,
        deep: int = 17,
    ):
        # Create an args namespace or dictionary to pass to kernel_error_model
        args = KerUncArgs(lmds, layers, deep)
        super().__init__(args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Inputs) -> Tensor:
        (
            image,
            psf,
        ) = inputs
        """Forward pass with sigmoid activation"""
        output = super().forward(image, psf)[-1]  # Get the final output
        return (self.sigmoid(output),)

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
