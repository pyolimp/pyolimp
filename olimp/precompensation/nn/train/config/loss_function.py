from __future__ import annotations
from typing import Annotated, Literal, Any, TypeAlias, Callable, Union
from collections.abc import Sequence
from .base import StrictModel
from pydantic import Field, confloat
from .....simulate import ApplyDistortion
from torch import Tensor


def _create_simple_loss(loss: Callable[[Tensor, Tensor], Tensor]):
    """
    Wrap loss function, that does not know about `Distortion`
    """

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
        from .....evaluation.loss.vae import vae_loss

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


class CVDSwinLossFunction(StrictModel):
    name: Literal["CVDSwinLoss"]
    lambda_ssim: Annotated[float, confloat(ge=0, le=1)] = 0.25
    global_points: int = 3000

    def load(self, _model: Any):
        from .....evaluation.loss.cvd_swin_loss import (
            CVDSwinLoss,
        )

        cbl = CVDSwinLoss(
            lambda_ssim=self.lambda_ssim,
            global_points=self.global_points,
        )

        def f(
            precompensated: Tensor,
            original_image: Tensor,
            distortion_fn: ApplyDistortion,
            extra: Sequence[Any],
        ) -> Tensor:

            assert precompensated.ndim == 4, precompensated.ndim
            return cbl(original_image, precompensated, distortion_fn)

        return f


class MSELossFunction(StrictModel):
    name: Literal["MSE"]

    def load(self, _model: Any):
        from .....evaluation.loss.mse import MSE

        mse = MSE()

        return _create_simple_loss(mse)


class RMSELossFunction(StrictModel):
    name: Literal["RMSE"]

    color_space: Literal["srgb", "lab", "prolab", "oklab"] = "srgb"

    def load(self, _model: Any):
        from .....evaluation.loss.rmse import RMSE

        rmse = RMSE(color_space=self.color_space)

        return _create_simple_loss(rmse)


class ProlabLossFunction(StrictModel):
    name: Literal["Prolab"]

    def load(self, _model: Any):
        from .....evaluation.loss.rmse import RMSE

        prolab = RMSE(color_space="prolab")

        return _create_simple_loss(prolab)


class PSNRLossFunction(StrictModel):
    name: Literal["PSNR"]

    def load(self, _model: Any):
        from .....evaluation.loss.psnr import PSNR

        psrn = PSNR()

        return _create_simple_loss(psrn)


class NRMSELossFunction(StrictModel):
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

    def load(self, _model: Any):
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


class ChromaticityDifferenceLossFunction(StrictModel):
    name: Literal["ChromaticityDifference"]
    color_space: Literal["lab", "prolab"]
    mode: Literal[
        "normal-color-vision",
        "color-vision-deficiency",
    ] = "normal-color-vision"

    def load(self, _model: Any):
        from .....evaluation.loss.chromaticity_difference import (
            ChromaticityDifference,
        )

        cd = ChromaticityDifference(self.color_space)

        if self.mode == "normal-color-vision":

            def loss_func(
                precompensated: Tensor,
                original_image: Tensor,
                distortion_fn: ApplyDistortion,
                extra: Sequence[Any],
            ) -> Tensor:
                return cd(precompensated, original_image, *extra)

        elif self.mode == "color-vision-deficiency":

            def loss_func(
                precompensated: Tensor,
                original_image: Tensor,
                distortion_fn: ApplyDistortion,
                extra: Sequence[Any],
            ) -> Tensor:
                return cd(
                    distortion_fn(precompensated),
                    distortion_fn(original_image),
                    *extra,
                )

        else:
            assert False

        return loss_func


class ContrastSimilarityLossFunction(StrictModel):
    name: Literal["contrast_similarity"]

    def load(self, _model: Any):
        from .....evaluation.loss.contrast_similarity import ContrastSimLoss

        return _create_simple_loss(ContrastSimLoss())


class VSILossFunction(StrictModel):
    name: Literal["VSI"]

    def load(self, _model: Any):
        from .....evaluation.loss.piq import VSILoss

        vsi = VSILoss()

        return _create_simple_loss(vsi)


class SOkLabLossFunction(StrictModel):
    name: Literal["SOkLab"]
    dpi: float
    distance_inch: float

    def load(self, _model: Any):
        from .....evaluation.loss.s_oklab import SOkLab

        s_oklab = SOkLab(dpi=self.dpi, distance_inch=self.distance_inch)

        return _create_simple_loss(s_oklab)


class HDRFLIPLossFunction(StrictModel):
    name: Literal["HDRFLIP"]

    def load(self, _model: Any):
        from .....evaluation.loss.flip import HDRFLIPLoss

        return _create_simple_loss(HDRFLIPLoss())


class LDRFLIPLossFunction(StrictModel):
    name: Literal["LDRFLIP"]

    def load(self, _model: Any):
        from .....evaluation.loss.flip import LDRFLIPLoss

        return _create_simple_loss(LDRFLIPLoss())


LossFunction = Annotated[
    ChromaticityDifferenceLossFunction
    | ContrastSimilarityLossFunction
    | CorrLossFunction
    | CVDSwinLossFunction
    | FSIMLossFunction
    | HDRFLIPLossFunction
    | LDRFLIPLossFunction
    | MultiScaleSSIMLossFunction
    | NRMSELossFunction
    | ProlabLossFunction
    | PSNRLossFunction
    | RMSLossFunction
    | SOkLabLossFunction
    | SSIMLossFunction
    | StressLossFunction
    | VaeLossFunction
    | VSILossFunction,
    Field(..., discriminator="name"),
]
