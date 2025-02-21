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
from ballfish.distribution import DistributionParams, create_distribution
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

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"
        datum.image = torch.fft.fftshift(datum.image)
        datum.image /= datum.image.sum(axis=(1, 2, 3), keepdim=True).view(
            -1, 1, 1, 1
        )
        return datum


class Float32Transform(Transformation):
    name = "float32"

    class Args(ArgDict):
        name: Literal["float32"]

    def __call__(self, datum: Datum, random: Random) -> Datum:
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

    def __call__(self, datum: Datum, random: Random) -> Datum:
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

        self._normalize = Normalize(mean, std, inplace=True)

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.source is not None
        self._normalize(datum.image)
        return datum


class WhitePoint(Transformation):
    name = "white_point"
    default_distribution: DistributionParams = {
        "name": "truncnorm",
        "a": 0.8,
        "b": 1.2,
        "mu": 1.0,
        "sigma": 0.25,
    }

    class Args(ArgDict):
        name: Literal["white_point"]
        l_factor: NotRequired[DistributionParams]
        m_factor: NotRequired[DistributionParams]
        s_factor: NotRequired[DistributionParams]

    def __init__(
        self,
        l_factor: DistributionParams = default_distribution,
        m_factor: DistributionParams = default_distribution,
        s_factor: DistributionParams = default_distribution,
    ) -> None:
        self._l = create_distribution(l_factor)
        self._m = create_distribution(m_factor)
        self._s = create_distribution(s_factor)
        from olimp.evaluation.cs.lms import LMS
        from olimp.evaluation.cs.srgb import sRGB

        def to_lms(color: torch.Tensor) -> torch.Tensor:
            return LMS().from_XYZ(sRGB().to_XYZ(color))

        def to_srgb(color: torch.Tensor) -> torch.Tensor:
            return sRGB().from_XYZ(LMS().to_XYZ(color))

        self._to_lms, self._to_srgb = to_lms, to_srgb

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None
        for image in datum.image:
            lms = self._to_lms(image)
            l = self._l(random)
            m = self._m(random)
            s = self._s(random)
            lms *= torch.tensor((l, m, s)).unsqueeze(1).unsqueeze(1)
            image[:] = self._to_srgb(lms)
        return datum


class ToneMappingHDRNet(Transformation):
    name = "tone_mapping_hdrnet"

    class Args(ArgDict):
        name: Literal["tone_mapping_hdrnet"]
        weights_path: str

    def __init__(self, weights_path: str) -> None:
        from .hdrnet import HDRnetModel

        self._model = HDRnetModel.from_path(weights_path)
        self._model.eval()

    def __call__(self, datum: Datum, random: Random):
        from torchvision.transforms.functional import resize

        assert datum.image is not None
        with torch.inference_mode():
            lowres, fullres = self._model.preprocess(datum.image)
            result = self._model(
                lowres.to(memory_format=torch.contiguous_format),
                fullres.to(memory_format=torch.contiguous_format),
            )

        datum.image = resize(result, datum.image.shape[-2:])
        return datum


BallfishTransforms = list[
    Annotated[
        ballfish.transformation.Args
        | CopyTransform.Args
        | Float32Transform.Args
        | NormalizeTransform.Args
        | PSFNormalize.Args
        | ToneMappingHDRNet.Args
        | WhitePoint.Args,
        Field(..., discriminator="name"),
    ]
]
