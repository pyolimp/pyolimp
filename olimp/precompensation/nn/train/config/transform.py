from __future__ import annotations
from typing import Annotated, Literal, TypedDict
import typing
from pydantic import Field, ConfigDict
from random import Random
import torch


# patch ballfish's typing to enable pydantic
from typing_extensions import NotRequired, TypedDict as TETypedDict

typing.TypedDict = TETypedDict  # monkeypatch for python 3.10

from ballfish.transformation import Datum, Transformation, ArgDict
import ballfish.transformation
import ballfish.distribution

typing.TypedDict = TypedDict  # undo monkeypatch

ballfish.transformation.NotRequired = NotRequired
ballfish.distribution.NotRequired = NotRequired


ArgDict.__pydantic_config__ = ConfigDict(extra="forbid", frozen=True)


class PSFNormalize(Transformation):
    name = "psf_normalize"

    class Args(ArgDict):
        name: Literal["psf_normalize"]

    def __call__(self, datum: Datum, random: Random):
        assert datum.image is not None, "missing datum.image"
        datum.image /= datum.image.sum(axis=(1, 2, 3), keepdim=True).view(
            -1, 1, 1, 1
        )
        return datum


class Float32Transform(Transformation):
    name = "float32"

    class Args(ArgDict):
        name: Literal["float32"]

    def __call__(self, datum: Datum, random: Random):
        assert datum.image is not None, "missing datum.image"
        datum.image = datum.image.to(torch.float32)
        return datum


class CopyTransform(Transformation):
    """
    Convenient method when no rasterization is needed.
    Meant for internal use only.
    """

    name = "_copy"

    class Args(ArgDict):
        name: Literal["_copy"]

    def __call__(self, datum: Datum, random: Random):
        assert datum.source is not None
        assert datum.image is None, "missing datum.image"
        datum.image = datum.source.clone()
        return datum


class NormalizeTransform(Transformation):
    name = "normalize"

    class Args(ArgDict):
        name: Literal["normalize"]
        mean: list[float]
        std: list[float]

    def __init__(self, mean: list[float], std: list[float]):
        from torchvision.transforms.v2 import Normalize

        self._normalize = Normalize(mean, std)

    def __call__(self, datum: Datum, random: Random):
        assert datum.source is not None
        datum.image = self._normalize(datum.image)
        return datum


BallfishTransforms = list[
    Annotated[
        ballfish.transformation.Args
        | PSFNormalize.Args
        | Float32Transform.Args
        | CopyTransform.Args
        | NormalizeTransform.Args,
        Field(..., discriminator="name"),
    ]
]
