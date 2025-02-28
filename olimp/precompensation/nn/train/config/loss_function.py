from __future__ import annotations
from typing import Annotated, Literal, Any, TypeAlias, Callable, Union
from collections.abc import Sequence
from .base import StrictModel
from pydantic import Field, confloat
from .....simulate import ApplyDistortion
from torch import Tensor


def _create_simple_loss(loss: Callable[[Tensor, Tensor], Tensor]):
    def f(
        precompensated: Tensor,
        original_image: Tensor,
        distortion_fn: ApplyDistortion,
        extra: Sequence[Any],
    ) -> Tensor:
        return loss(distortion_fn(precompensated), original_image, *extra)

    return f


class VaeLossFunction(StrictModel):
    name: Literal["Vae"]

    def load(self, model: Any):
        from .....evaluation.loss import vae_loss

        assert type(model).__name__ in (
            "VAE",
            "CVAE",
            "UNETVAE",
        ), f"Vae loss only work with (C,UNET)Vae model, not {model}"

        def f(
            precompensated: Tensor,
            original_image: Tensor,
            distortion_fn: ApplyDistortion,
            extra: Sequence[Any],
        ) -> Tensor:
            return vae_loss(
                distortion_fn(precompensated), original_image, *extra
            )

        return f


Degree: TypeAlias = Literal[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class ColorBlindnessLossFunction(StrictModel):
    name: Literal["ColorBlindnessLoss"]
    degree: Degree = 100
    lambda_ssim: Annotated[float, confloat(ge=0, le=1)] = 0.25
    global_points: int = 3000

    def load(self, _model: Any):
        from .....evaluation.loss.color_blindness_loss import (
            ColorBlindnessLoss,
        )
        from .....simulate.color_blindness_distortion import (
            ColorBlindnessDistortion,
        )

        cbl = ColorBlindnessLoss(
            lambda_ssim=self.lambda_ssim,
            global_points=self.global_points,
        )

        def f(
            model_output: list[Tensor],
            datums: list[Tensor],
            distortions: list[type[Distortion]],
        ) -> Tensor:
            (cbd,) = distortions
            assert isinstance(cbd, ColorBlindnessDistortion)
            cbl.set_cb_type(cbd.blindness_type, self.degree)
            (image,) = datums
            assert image.ndim == 4, image.ndim
            (precompensated,) = model_output
            assert precompensated.ndim == 4, precompensated.ndim
            return cbl(image, precompensated)

        return f


class MSELossFucntion(StrictModel):
    name: Literal["MSE"]

    def load(self, _model: Any):
        from .....evaluation.loss.mse import MSE

        mse = MSE()

        return _create_simple_loss(mse)


class PSNRLossFunction(StrictModel):
    name: Literal["PSNR"]

    def load(self, _model: Any):
        from .....evaluation.loss.psnr import PSNR

        psrn = PSNR()

        return _create_simple_loss(psrn)


class NRMSELossFuction(StrictModel):
    name: Literal["NRMSE"]

    invert: bool = False

    def load(self, _model: Any):
        from .....evaluation.loss.nrmse import NormalizedRootMSE

        nrmse = NormalizedRootMSE(invert=self.invert)

        return _create_simple_loss(nrmse)


class StressLossFunction(StrictModel):
    name: Literal["STRESS"]

    def load(self, _model: Any):
        from .....evaluation.loss.stress import STRESS

        stress = STRESS()

        return _create_simple_loss(stress)


class CorrLossFunction(StrictModel):
    name: Literal["CORR"]

    def laod(self, _model: Any):
        from .....evaluation.loss.corr import Correlation

        corr = Correlation()

        return _create_simple_loss(corr)


class SSIMLossFunction(StrictModel):
    name: Literal["SSIM"]

    kernel_size: int = 11
    kernel_sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03

    def load(self, _model: Any):
        from .....evaluation.loss.piq import SSIMLoss

        ssim = SSIMLoss(
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

        return _create_simple_loss(ssim)


class MultiScaleSSIMLossFunction(StrictModel):
    name: Literal["MS_SSIM"]

    kernel_size: int = 11
    kernel_sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03

    def load(self, _model: Any):
        from .....evaluation.loss.piq import MultiScaleSSIMLoss

        ms_ssim = MultiScaleSSIMLoss(
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

        return _create_simple_loss(ms_ssim)


class FSIMLossFunction(StrictModel):
    name: Literal["FSIM"]

    reduction: Literal["none", "mean", "sum"] = "mean"
    data_range: Union[int, float] = 1.0
    chromatic: bool = True
    scales: int = 4
    orientations: int = 4
    min_length: int = 6
    mult: int = 2
    sigma_f: float = 0.55
    delta_theta: float = 1.2
    k: float = 2.0

    def load(self, _model: Any):
        from .....evaluation.loss.piq import FSIMLoss

        fsim = FSIMLoss(
            reduction=self.reduction,
            data_range=self.data_range,
            chromatic=self.chromatic,
            scales=self.scales,
            orientations=self.orientations,
            min_length=self.min_length,
            mult=self.mult,
            sigma_f=self.sigma_f,
            delta_theta=self.delta_theta,
            k=self.k,
        )

        return _create_simple_loss(fsim)


class RMSLossFunction(StrictModel):
    name: Literal["RMS"]
    color_space: Literal["lab", "prolab"]

    n_pixel_neighbors: int = 1000
    step: int = 10
    sigma_rate: float = 0.25

    def load(self, _model: Any):
        from .....evaluation.loss.rms import RMS

        rms = RMS(
            self.color_space,
            n_pixel_neighbors=self.n_pixel_neighbors,
            step=self.step,
            sigma_rate=self.sigma_rate,
        )

        return _create_simple_loss(rms)


class LabLossFunction(StrictModel):
    name: Literal["Lab"]
    lightness_weight: int = 0

    def load(self, _model: Any):
        from .....evaluation.loss.chromaticity_difference import Lab

        lab = Lab(
            lightness_weight=self.lightness_weight,
        )

        return _create_simple_loss(lab)


class ProLabLossFunction(StrictModel):
    name: Literal["ProLab"]
    lightness_weight: int = 0

    def load(self, _model: Any):
        from .....evaluation.loss.chromaticity_difference import ProLab

        prolab = ProLab(
            lightness_weight=self.lightness_weight,
        )

        return _create_simple_loss(prolab)


class VSILossFunction(StrictModel):
    name: Literal["VSI"]

    def load(self, _model: Any):
        from .....evaluation.loss.vsi import VSI

        vsi = VSI()

        return _create_simple_loss(vsi)


class SOkLabLossFunction(StrictModel):
    name: Literal["SOkLab"]

    def load(self, _model: Any):
        from .....evaluation.loss.s_oklab import SOkLab

        s_oklab = SOkLab()

        return _create_simple_loss(s_oklab)


LossFunction = Annotated[
    VaeLossFunction
    | ColorBlindnessLossFunction
    | MultiScaleSSIMLossFunction
    | RMSLossFunction
    | VSILossFunction,
    Field(..., discriminator="name"),
]
