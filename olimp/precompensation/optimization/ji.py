from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
from torch import Tensor
from olimp.evaluation.loss.mse import MSE
from olimp.precompensation.basic.huang import huang

from typing import NamedTuple, TypedDict
from olimp.processing import fft_conv


class DebugInfo(TypedDict):
    loss_step: list[float]
    precomp: Tensor


# Parameter storage class
class JiParameters(NamedTuple):
    rgb: bool = False
    lr: float = 1e-3
    lr_m: float = 0.5
    m0: float = 1.0
    gap: float = 0.01
    gap_iter: float = 0.001
    ibf_param: float = 0.01
    partition_step: float = 0.01
    alpha: float = 0.0
    num_of_iter: int = 100
    loss_func: Callable[[Tensor, Tensor], Tensor] | None = None
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None  # pass dict to log


def linear_normalise(image: Tensor) -> Tensor:
    return (image - torch.min(image)) / (torch.max(image) - torch.min(image))


def histogram_maximum(image: Tensor) -> float:
    img = linear_normalise(image)
    num_of_bins = 1000
    hist = torch.histc(img, bins=num_of_bins, min=0, max=1)
    return torch.argmax(hist) / num_of_bins


def bezier_curve(
    mode: float,
    m: Tensor,
    tau_plus: Tensor,
    tau_minus: Tensor,
    partition: Tensor,
) -> tuple[Tensor, Tensor]:
    theta = torch.arctan(1 / m)
    Q_minus = [
        mode - tau_minus * torch.sin(theta),
        mode - tau_minus * torch.cos(theta),
    ]
    Q_plus = [
        mode + tau_plus * torch.sin(theta),
        mode + tau_plus * torch.cos(theta),
    ]
    P = [mode, mode]
    t = partition

    Bx_minus = (
        # torch.square(1 - t) * 0
        +2 * t * (1 - t) * Q_minus[0]
        + torch.square(t) * P[0]
    )
    By_minus = (
        # torch.square(1 - t) * 0
        +2 * t * (1 - t) * Q_minus[1]
        + torch.square(t) * P[1]
    )

    Bx_plus = (
        torch.square(1 - t) * P[0]
        + 2 * t * (1 - t) * Q_plus[0]
        + torch.square(t) * 1
    )
    By_plus = (
        torch.square(1 - t) * P[1]
        + 2 * t * (1 - t) * Q_plus[1]
        + torch.square(t) * 1
    )

    Bx = torch.cat([Bx_minus, Bx_plus])
    By = torch.cat([By_minus, By_plus])

    return Bx, By


def mapping_function(
    image: Tensor,
    m: Tensor,
    tau_plus: Tensor,
    tau_minus: Tensor,
    mode: float,
    partition: Tensor,
) -> Tensor:
    Bx, By = bezier_curve(mode, m, tau_plus, tau_minus, partition)
    length = Bx.size(dim=0)
    indices = torch.searchsorted(Bx, image, right=False)
    indices = torch.clamp(indices, 0, length - 1)
    return By[indices]


def ji(image: Tensor, psf: Tensor, params: JiParameters) -> Tensor:
    partition = torch.arange(0, 1, params.partition_step)
    ibf = huang(image, psf, params.ibf_param)
    ibf = linear_normalise(ibf)
    mode = histogram_maximum(ibf)

    m0 = torch.tensor([params.m0], dtype=torch.float32)
    tau_plus0 = torch.tensor([0.25], dtype=torch.float32)
    tau_minus0 = torch.tensor([0.25], dtype=torch.float32)

    def loss_func(x: Tensor, y: Tensor, m: Tensor) -> Tensor:
        if params.loss_func is None:
            loss_ji = MSE()
            return loss_ji(x, y) + params.alpha / m
        else:
            return params.loss_func(x, y)

    prev_total_loss = torch.tensor(float("inf"))

    loss_step: list[float] = []
    if params.debug is not None:
        params.debug["loss_step"] = loss_step

    for i in range(params.num_of_iter):
        if params.progress is not None:
            params.progress(i / params.num_of_iter)

        # Solving tau subproblem with fixed m
        prev_loss_tau = torch.tensor(float("inf"))

        tau_plus = tau_plus0.clone().detach().requires_grad_()
        tau_minus = tau_minus0.clone().detach().requires_grad_()
        optimizer_tau = torch.optim.Adam([tau_plus, tau_minus], lr=params.lr)

        for _ in range(1000):
            optimizer_tau.zero_grad()

            mapped = mapping_function(
                ibf,
                m0,
                tau_plus,
                tau_minus,
                mode,
                partition,
            )

            mapped_conv = fft_conv(mapped, psf)
            loss = loss_func(mapped_conv, image, m0)

            loss.backward(retain_graph=True)
            optimizer_tau.step()

            if torch.abs(prev_loss_tau - loss).item() < params.gap:
                break

            prev_loss_tau = loss

        # Updating params

        theta = torch.arctan(1 / m0)

        tau_plus0 = torch.clamp(
            tau_plus.clone().detach(),
            0,
            ((1 - mode) / torch.cos(theta.clone())).item(),
        )
        tau_minus0 = torch.clamp(
            tau_minus.clone().detach(),
            0,
            (mode / torch.cos(theta.clone())).item(),
        )

        # Solving m-subproblem with fixed tau
        prev_loss_m = torch.tensor(float("inf"))

        m = m0.clone().detach().requires_grad_()

        optimizer_m = torch.optim.Adam([m], lr=params.lr_m)

        for _ in range(1000):
            optimizer_m.zero_grad()

            mapped = mapping_function(
                ibf,
                m,
                tau_plus0,
                tau_minus0,
                mode,
                partition,
            )

            mapped_conv = fft_conv(mapped, psf)
            loss = loss_func(mapped_conv, image, m)

            loss.backward(retain_graph=True)
            optimizer_m.step()

            if torch.abs(prev_loss_m - loss).item() < params.gap:
                break

            prev_loss_m = loss

        # Update m
        m0 = m.detach()

        loss_step.append(prev_loss_m.item())

        if abs(prev_total_loss - prev_loss_m.item()) < params.gap_iter:
            break

        prev_total_loss = prev_loss_m.item()

    if params.progress is not None:
        params.progress(1.0)

    return mapping_function(ibf, m0, tau_plus0, tau_minus0, mode, partition)


def _demo():
    from .._demo import demo
    from olimp.evaluation.loss.piq import MultiScaleSSIMLoss

    def demo_ji(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return ji(
            image,
            psf,
            JiParameters(
                progress=progress, alpha=1, loss_func=MultiScaleSSIMLoss()
            ),
        )

    demo("Ji", demo_ji)


if __name__ == "__main__":
    _demo()
