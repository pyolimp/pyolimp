from __future__ import annotations
from typing import Annotated, Literal, Any
from .base import StrictModel
from pydantic import Field
from olimp.processing import conv
from torch import Tensor
import torch


class VaeLossFunction(StrictModel):
    name: Literal["Vae"]

    def load(self, _model: Any):
        from .....evaluation.loss import vae_loss

        def f(model_output: list[Tensor], datums: list[Tensor]) -> Tensor:
            image, psf = datums
            precompensated, *args = model_output
            retinal_precompensated = conv(
                precompensated.to(torch.float32).clip(0, 1), psf
            )
            loss = vae_loss(retinal_precompensated, image, *args)
            return loss

        return f


LossFunction = Annotated[
    VaeLossFunction,
    Field(..., discriminator="name"),
]
