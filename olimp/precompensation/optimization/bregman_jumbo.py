from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
import torch.nn.functional as F
from olimp.processing import fft_conv


class DebugInfo(TypedDict):
    loss_step: list[float]
    p: torch.Tensor


class BregmanJumboParameters(NamedTuple):
    lr: float = 1e-2
    Lambda: float = 10.0
    c_high: float = 0.95
    c_low: float = 1.0 - c_high
    gap: float = 0.0001
    gap_breg: float = 0.01
    gamma: float = 1e-3
    beta: float = 1.0
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None  # pass dict to log


def _grad(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    im_bg = F.pad(input=image, pad=(1, 1, 1, 1))

    x1 = im_bg[..., 1:-1, :-2]
    y1 = im_bg[..., :-2, 1:-1]

    dx = image - x1
    dy = image - y1

    return dx, dy


def _clip(image: torch.Tensor) -> torch.Tensor:
    return image.clip(0.0, 1.0)


def bregman_jumbo(
    image: torch.Tensor,
    psf: torch.Tensor,
    parameters: BregmanJumboParameters = BregmanJumboParameters(),
) -> torch.Tensor:
    """
    .. image:: ../../_static/bregman_jumbo.svg
       :class: full-width
    """
    # Parameters
    lr = parameters.lr
    gap = parameters.gap
    gap_breg = parameters.gap_breg

    # Constant
    beta = parameters.beta
    lam = parameters.Lambda
    gam = parameters.gamma

    # Preparing image
    t = image * (parameters.c_high - parameters.c_low) + parameters.c_low

    # Calculating loss
    loss_step: list[float] = []
    if parameters.debug is not None:
        parameters.debug["loss_step"] = loss_step

    # Optimization parameters on k - 1 iteration
    b_z, b_x, b_y = (
        torch.zeros_like(image, requires_grad=True),
        torch.zeros_like(image, requires_grad=True),
        torch.zeros_like(image, requires_grad=True),
    )
    p_prev = t.clone()
    g_x, g_y = _grad(p_prev)
    z = t.clone()

    progress: Callable[[float], None] = (
        lambda val: parameters.progress and parameters.progress(val)
    )
    progress(0.1)

    for k in range(20):
        # Optimization parameters
        self_p = p_prev.clone().detach().requires_grad_(True)

        # Optimizators
        optimizer_p = torch.optim.Adam([self_p], lr=lr)

        # Solving g subproblem
        gd_x, gd_y = _grad(p_prev)
        sgn_x, sgn_y = torch.sign(gd_x + b_x), torch.sign(gd_y + b_y)
        # grad_norm = torch.hypot(gd_x + b_x, gd_y + b_y)
        grad_norm = torch.sqrt(
            torch.pow(gd_x + b_x, 2) + torch.pow(gd_y + b_y, 2)
        )
        mx = torch.maximum(grad_norm - 1 / gam, torch.zeros(image.shape))
        self_g_x, self_g_y = torch.mul(sgn_x, mx), torch.mul(sgn_y, mx)

        u = fft_conv(p_prev, psf) + b_z
        self_z = torch.minimum(
            torch.maximum(t, u - lam / beta), u + lam / beta
        )

        # Solving p subproblem
        prev_loss = torch.tensor(float("inf"))

        for i in range(5000):
            optimizer_p.zero_grad()
            p_clipped = _clip(self_p)
            e = z - fft_conv(p_clipped, psf) - b_z
            gd_x, gd_y = _grad(p_clipped)

            # func_l2 = torch.sum(torch.square(e))
            func_l2 = torch.sum(torch.square(e))
            func_breg = torch.sum(
                torch.square(gd_x - g_x - b_x) + torch.square(gd_y - g_y - b_y)
            )
            loss_p = ((beta / 2) * func_l2) + ((gam / 2) * func_breg)

            loss_p.backward(retain_graph=True)
            optimizer_p.step()

            if torch.abs(prev_loss - loss_p) < gap:
                break

            prev_loss = loss_p
        if parameters.debug is not None:
            parameters.debug["p"] = self_p

        # Bregman iteration stop criteria
        criteria = torch.sum(torch.abs(p_prev - _clip(self_p))) / (
            self_p.shape[-1] * self_p.shape[-2]
        )
        loss_step.append(criteria.item())
        if len(loss_step) > 1 and loss_step[-2] * 1.1 < loss_step[-1]:
            return p_prev
        if criteria < gap_breg:
            break

        # Updating Bregman parameters
        p_prev = _clip((self_p).clone())
        z = (self_z).clone()
        g_x, g_y = (self_g_x).clone(), (self_g_y).clone()

        grad_x, grad_y = _grad(p_prev)
        b_x, b_y = b_x + grad_x - g_x, b_y + grad_y - g_y
        b_z = b_z + fft_conv(p_prev, psf) - z
        progress(0.2 + (k / 19) * 0.8)
    progress(1.0)
    return self_p


def _demo():
    from .._demo import demo

    def demo_bregman_jumbo(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        return bregman_jumbo(
            image, psf, BregmanJumboParameters(progress=progress)
        )

    demo("Bregman Jumbo", demo_bregman_jumbo)


if __name__ == "__main__":
    _demo()
