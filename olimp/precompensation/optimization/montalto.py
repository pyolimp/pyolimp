from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from olimp.processing import fft_conv


class DebugInfo(TypedDict):
    loss_step: list[float]
    precomp: Tensor


class MontaltoParameters(NamedTuple):
    lr: float = 1e-2
    theta: float = 1e-6
    tau: float = 2e-5
    Lambda: float = 65.0
    c_high: float = 1.0
    c_low: float = 1 - c_high
    gap: float = 0.01
    loss_func: Callable[[Tensor, Tensor], Tensor] | None = None
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None  # pass dict to log


def _tv_func(image: Tensor) -> Tensor:
    im_bg = F.pad(input=image, pad=(1, 1, 1, 1))

    x1 = im_bg[..., 1:-1, :-2]
    y1 = im_bg[..., :-2, 1:-1]

    dx = torch.abs(image - x1)
    dy = torch.abs(image - y1)

    grad_image = dx + dy

    return grad_image


def montalto(
    image: Tensor,
    psf: Tensor,
    parameters: MontaltoParameters = MontaltoParameters(),
) -> Tensor:
    """
    .. image:: ../../_static/montalto.svg
       :class: full-width
    """
    # Parametrs
    lr = parameters.lr

    # Constant
    theta = parameters.theta
    tau = parameters.tau
    Lambda = parameters.Lambda

    c_high = parameters.c_high
    c_low = parameters.c_low

    # Preparing image
    t = image * (c_high - c_low) + c_low

    # Calculating loss
    prev_loss = torch.tensor(float("inf"))
    precomp = t.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([precomp], lr=lr)
    loss_step = []
    if parameters.debug is not None:
        parameters.debug["loss_step"] = loss_step

    for i in range(5000):
        if parameters.progress is not None:
            parameters.progress(i / 5000)
        optimizer.zero_grad()

        if parameters.loss_func is None:
            e = fft_conv(precomp, psf) - t

            func_l1 = torch.sum(_tv_func(precomp))
            func_l2 = torch.linalg.norm(e.flatten())
            func_borders = torch.sum(
                torch.exp(-Lambda * precomp)
                + torch.exp(-Lambda * (1 - precomp))
            )
            loss = func_l2 + (theta * func_l1) + (tau * func_borders)
        else:
            retinal_precomp = fft_conv(precomp.clip(0, 1), psf)
            loss = parameters.loss_func(
                retinal_precomp.clip(0, 1), t.clip(0, 1)
            )

        loss_step.append(loss.item())
        loss.backward()
        optimizer.step()

        if parameters.debug is not None:
            parameters.debug["precomp"] = precomp

        if torch.abs(prev_loss - loss).item() < parameters.gap:
            break

        prev_loss = loss
    if parameters.progress is not None:
        parameters.progress(1.0)

    return precomp.clip(0, 1)


def _demo():
    from .._demo import demo

    def demo_montalto(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return montalto(image, psf, MontaltoParameters(progress=progress))

    demo("Montalto", demo_montalto)


if __name__ == "__main__":
    _demo()
