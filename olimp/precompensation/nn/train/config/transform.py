from __future__ import annotations
from typing import Annotated, Literal, Callable
from pydantic import Field
import torch
from torch import Tensor
from torchvision.transforms.v2 import (
    Resize,
    Grayscale,
    Normalize,
    InterpolationMode,
)
from .base import StrictModel


class TransformsTransform(StrictModel):
    pass


class ResizeTransform(TransformsTransform):
    name: Literal["Resize"]
    size: list[int] = [512, 512]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR

    def transform(self):
        return Resize(self.size, interpolation=self.interpolation)


class GrayscaleTransform(TransformsTransform):
    name: Literal["Grayscale"]
    num_output_channels: int = 1

    def transform(self):
        return Grayscale(self.num_output_channels)


class DivideTransform(TransformsTransform):
    name: Literal["Divide"]
    value: float = 255.0

    def transform(self):
        return self

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor / self.value


class NormalizeTransform(TransformsTransform):
    name: Literal["Normalize"]
    mean: list[float]
    std: list[float]

    def transform(self):
        return Normalize(self.mean, self.std)


class Float32Transform(TransformsTransform):
    name: Literal["Float32"]

    def transform(self) -> Callable[[Tensor], Tensor]:
        return lambda t: t.to(torch.float32)


class PSFNormalizeTransform(TransformsTransform):
    name: Literal["PSFNormalize"]

    def transform(self) -> Callable[[Tensor], Tensor]:
        return self

    def __call__(self, psf: Tensor) -> Tensor:
        psf /= psf.sum(axis=(1, 2, 3), keepdim=True).view(-1, 1, 1, 1)
        return psf


Transforms = list[
    Annotated[
        ResizeTransform
        | GrayscaleTransform
        | DivideTransform
        | NormalizeTransform
        | PSFNormalizeTransform
        | Float32Transform,
        Field(discriminator="name"),
    ]
]
