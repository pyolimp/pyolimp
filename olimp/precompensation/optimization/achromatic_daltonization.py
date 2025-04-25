from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
from torch import Tensor

from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion
from olimp.simulate import ApplyDistortion
from olimp.processing import quantile_clip


def _grad(image: Tensor) -> tuple[Tensor, Tensor]:
    dx = image - torch.roll(image, shifts=1, dims=-1)
    dy = image - torch.roll(image, shifts=1, dims=-2)
    return dx, dy


class M1Loss:
    def __call__(
        self,
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        average_value: float = 0.8,
        eps: float = 0.015,
    ) -> Callable[[Tensor], Tensor]:
        distortion_apply = distortion()
        grad_x, grad_y = _grad(image)
        norm2_x = torch.sum(grad_x**2, dim=1)
        norm2_y = torch.sum(grad_y**2, dim=1)

        def loss(weight_map: Tensor) -> Tensor:
            simulated = distortion_apply(image * weight_map)
            sim_grad_x, sim_grad_y = _grad(simulated)
            sim_norm2_x = torch.sum(sim_grad_x**2, dim=1)
            sim_norm2_y = torch.sum(sim_grad_y**2, dim=1)

            mat_err = torch.pow(
                torch.sqrt(sim_norm2_x + 1e-32) - torch.sqrt(norm2_x + 1e-32),
                2,
            ) / (norm2_x + eps**2) + torch.pow(
                torch.sqrt(sim_norm2_y + 1e-32) - torch.sqrt(norm2_y + 1e-32),
                2,
            ) / (
                norm2_y + eps**2
            )
            return mat_err.mean(dim=(-1, -2)).sum()

        return loss


class M2Loss:
    @staticmethod
    def _get_dw(
        grad: Tensor,
        smooth: Tensor,
        sign: Tensor,
        average_value: float,
        distortion_apply: ApplyDistortion,
    ) -> Tensor:
        """
        Find direction and its module.
        Solve simple quadratic equation.
        """
        a = distortion_apply(smooth)
        b = average_value * distortion_apply(grad)
        c = torch.sum(grad**2, dim=1)

        D_pt2 = 4 * torch.sum(a**2, dim=1) * (torch.sum(b**2, dim=1) - c)
        D_pt1 = 2 * torch.sum(a * b, dim=1)
        D = D_pt1**2 - D_pt2

        print(f"D < 0: {(D < 0).sum()}")
        print(f"DMax: {(D[D <= 0]).min()}")

        D[D < 0] = 0

        dw1 = (-D_pt1 + torch.sqrt(D)) / (2 * torch.sum(a**2, dim=1) + 1e-10)
        dw2 = (-D_pt1 - torch.sqrt(D)) / (2 * torch.sum(a**2, dim=1) + 1e-10)

        mask = sign > 0
        return torch.maximum(dw1, dw2) * mask + torch.minimum(dw1, dw2) * (
            ~mask
        )

    def __call__(
        self,
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        average_value: float = 0.8,
        eps: float = 0.015,
    ) -> Callable[[Tensor], Tensor]:
        distortion_apply = distortion()
        sign_x, sign_y = _grad(image.mean(dim=1))
        grad_x, grad_y = _grad(image)
        smooth_x = (image + torch.roll(image, 1, dims=-1)) / 2
        smooth_y = (image + torch.roll(image, 1, dims=-2)) / 2

        dw_x = self._get_dw(
            grad_x, smooth_x, sign_x, average_value, distortion_apply
        )
        dw_y = self._get_dw(
            grad_y, smooth_y, sign_y, average_value, distortion_apply
        )

        def loss(weight_map: Tensor) -> Tensor:
            dx, dy = _grad(weight_map)
            mat_err = (dx - dw_x) ** 2 / (dw_x**2 + eps**2) + (
                dy - dw_y
            ) ** 2 / (dw_y**2 + eps**2)
            return mat_err.mean(dim=(-1, -2)).sum()

        return loss


class DebugInfo(TypedDict):
    loss_step: list[float]
    weight_map: Tensor


class ADParameters(NamedTuple):
    average_value: float = 0.8
    eps: float = 0.015
    iterations: int = 20000
    lr: float = 1e-4
    gap: float = 1e-8
    loss_func: Callable[
        [Tensor, ColorBlindnessDistortion, float, float],
        Callable[[Tensor], Tensor],
    ] = M2Loss()
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None


def achromatic_daltonization(
    image: Tensor,
    distortion: ColorBlindnessDistortion,
    parameters: ADParameters = ADParameters(),
) -> Tensor:
    """Compute Dalt algorithm using watermark approach on set of images"""
    assert image.ndim == 4, image.shape

    lr = parameters.lr
    gap = parameters.gap
    iterations = parameters.iterations

    b, c, h, w = image.shape
    weight_map = torch.ones((b, h, w), requires_grad=True, device=image.device)

    optimizer = torch.optim.Adam([weight_map], lr=lr)
    loss_step = []

    if parameters.debug is not None:
        parameters.debug["loss_step"] = loss_step

    loss_func = parameters.loss_func(
        image, distortion, parameters.average_value, parameters.eps
    )
    prev_loss = torch.tensor(float("inf"))

    for i in range(iterations):
        if parameters.progress is not None:
            parameters.progress(i / iterations)

        optimizer.zero_grad()
        loss = loss_func(weight_map)
        loss_step.append(loss.item())
        loss.backward()
        optimizer.step()

        if parameters.debug is not None:
            parameters.debug["weight_map"] = weight_map

        if torch.abs(prev_loss - loss).item() < gap:
            break
        prev_loss = loss

    if parameters.progress is not None:
        parameters.progress(1.0)

    return quantile_clip(image * weight_map)


def _demo():
    from .._demo_cvd import demo

    def demo_achromatic_daltonization(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> tuple[Tensor]:

        return (
            achromatic_daltonization(
                image,
                distortion,
                ADParameters(progress=progress, loss_func=M1Loss()),
            ),
        )

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo(
        "Achromatic Daltonization",
        demo_achromatic_daltonization,
        distortion=distortion,
    )


if __name__ == "__main__":
    _demo()
