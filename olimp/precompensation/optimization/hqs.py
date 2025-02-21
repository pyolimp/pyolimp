from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
from olimp.processing import fft_conv


class DebugInfo(TypedDict):
    loss_step: list[float]
    p: torch.Tensor


class HQSParameters(NamedTuple):
    lr: float = 1e-2
    c_high: float = 0.95
    c_low: float = 1.0 - c_high
    gap: float = 1e-3
    gap_on_iteration: float = 1e-4
    mu: float = 1e-2
    beta: float = 1e-7
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None


def _grad_operators(shape: torch.Size):
    Dx = torch.zeros(shape)
    Dx[0][0][0][0] = 1
    Dx[0][0][0][1] = -1

    Dy = torch.zeros(shape)
    Dy[0][0][0][0] = 1
    Dy[0][0][1][0] = -1

    return (Dx, Dy)


def _grad(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Dx, Dy = _grad_operators(image.shape)
    image_dx = fft_conv(image, Dx)
    image_dy = fft_conv(image, Dy)
    return image_dx, image_dy


def _w_subproblem_solver(
    p: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    Dx, Dy = _grad_operators(p.shape)

    du_x = fft_conv(p, Dx)
    du_y = fft_conv(p, Dy)

    wx = torch.maximum(
        torch.sum(torch.abs(du_x)) - 1 / beta, torch.zeros(p.shape)
    ) * torch.sign(du_x)
    wy = torch.maximum(
        torch.sum(torch.abs(du_y)) - 1 / beta, torch.zeros(p.shape)
    ) * torch.sign(du_y)

    return (wx, wy)


def _clip(image: torch.Tensor) -> torch.Tensor:
    return image.clip(0.0, 1.0)


def hqs(
    image: torch.Tensor,
    psf: torch.Tensor,
    parameters: HQSParameters = HQSParameters(),
) -> torch.Tensor:
    """
    .. image:: ../../_static/hqs.svg
       :class: full-width
    """
    # Parameters
    lr = parameters.lr
    gap = parameters.gap
    gap_iter = parameters.gap_on_iteration

    # Constant
    beta = parameters.beta
    mu = parameters.mu

    # Preparing image
    t = image * (parameters.c_high - parameters.c_low) + parameters.c_low

    # Calculating loss
    loss_step: list[float] = []
    if parameters.debug is not None:
        parameters.debug["loss_step"] = loss_step

    progress: Callable[[float], None] = (
        lambda val: parameters.progress and parameters.progress(val)
    )
    progress(0.1)

    p0 = t

    for k in range(5000):
        # Solving w-subproblem
        wx, wy = _w_subproblem_solver(p0, beta)

        # Solving p-subproblem
        p = torch.tensor(p0.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([p], lr=lr)
        prev_loss_iter = torch.tensor(float("inf"))

        for i in range(1000):
            p_lim = _clip(p)
            p_dx, p_dy = _grad(p_lim)
            retinal = fft_conv(p_lim, psf)

            # Loss function
            func1 = torch.sum(torch.square(retinal - t))
            func2 = torch.sum(
                torch.square(wx - p_dx) + torch.square(wy - p_dy)
            )
            loss_func = (mu / 2) * func1 + (beta / 2) * func2

            # Optimization
            optimizer.zero_grad()
            loss_func.backward(retain_graph=True)
            optimizer.step()

            if torch.abs(prev_loss_iter - loss_func).item() < gap:
                break

            prev_loss_iter = loss_func

        if parameters.debug is not None:
            parameters.debug["p"] = _clip(p)

        criteria = torch.sum(torch.abs(p0 - _clip(p))) / (
            p0.shape[-1] * p0.shape[-2]
        )
        loss_step.append(criteria.item())
        if len(loss_step) > 1 and loss_step[-2] * 1.1 < loss_step[-1]:
            return p0
        if criteria < gap_iter:
            break

        p0 = _clip(p)
        progress(0.2 + (k / 19) * 0.8)

    progress(1.0)
    return p0


def _demo():
    from .._demo import demo

    def demo_hqs(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        return hqs(image, psf, HQSParameters(progress=progress))

    demo("Half-Quadratic", demo_hqs)


if __name__ == "__main__":
    _demo()
