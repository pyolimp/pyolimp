from __future__ import annotations
from typing import NamedTuple, TypedDict, Literal, Callable
import torch
from torch import Tensor
from torch import nn
from torch import fft
from .solvers import W_solver, Z_solver
from .filters import grad, sobel
from .loss import PreCompensationLoss


class DebugInfo(TypedDict):
    loss_step: list[float]
    p: Tensor


class HQS_PC_Parameters(NamedTuple):
    """Classic HQS-PC parameters"""

    # optimization prms
    max_iter: int = 1000
    epsilon: float | None = 1e-4
    z0_type: Literal["source", "zeros", "ones", "random"] = "source"
    verbose: bool = False
    # loss prms
    filters: Tensor | Literal["grad", "sobel"] = "grad"
    theta: float = 1e-6
    r_norm_type: Literal["l1", "l2"] = "l1"
    # parameters
    beta: float | list[float] = 1e-3
    gamma: float = 1e-6
    progress: Callable[[float], None] | None = None
    debug: DebugInfo | None = None


def init_var(
    source: Tensor, init_type: Literal["source", "zeros", "ones", "random"]
) -> Tensor:
    """Initialize variable"""
    if init_type == "source":
        return source.clone()
    elif init_type == "zeros":
        return torch.zeros_like(source)
    elif init_type == "ones":
        return torch.ones_like(source)
    elif init_type == "random":
        return torch.randn_like(source)
    else:
        raise ValueError(f"Unexpected initial variable type: {init_type}")


def hqs_pc(
    t: Tensor, k: Tensor, parameters: HQS_PC_Parameters = HQS_PC_Parameters()
) -> torch.Tensor:
    """Compute HQS-PC for image `t` and kernel `k`"""
    batch_size = t.size(0)

    # prepare solvers
    w_solver = W_solver(parameters.r_norm_type)
    z_solver = Z_solver()

    # prepare parameters
    beta_tmp = parameters.beta
    if isinstance(beta_tmp, float):
        beta = beta_tmp * torch.ones(
            batch_size, parameters.max_iter, device=t.device
        )
    else:
        beta = torch.tensor(beta_tmp, device=t.device)

    gamma = parameters.gamma * torch.ones(batch_size, device=t.device)
    eps = parameters.epsilon

    filter_tmp = parameters.filters
    if isinstance(filter_tmp, str):
        if filter_tmp == "grad":
            filters = grad(t.shape).to(t.device)
        elif filter_tmp == "sobel":
            filters = sobel(t.shape).to(t.device)
        else:
            raise ValueError(f"Unexpected filter type: {filter_tmp}")
    else:
        filters = filter_tmp

    # fft of blur kernel
    kernel_fft = fft.fft2(k)

    # fft of filters
    filters_fft = fft.fft2(filters)

    # fft of product K^T*t
    kt_fft = torch.conj(kernel_fft) * fft.fft2(t)

    # initial z value
    z0 = init_var(t, parameters.z0_type)
    p_new = z0.clone()

    # iteration loop
    loss = PreCompensationLoss(
        theta=parameters.theta,
        filters=filters,
        regularization=parameters.r_norm_type,
        reduction="mean",
    )
    loss_step = []
    if parameters.debug is not None:
        parameters.debug["loss_step"] = loss_step

    progress: Callable[[float], None] = (
        lambda val: parameters.progress and parameters.progress(val)
    )
    progress(0.1)

    for i in range(parameters.max_iter):
        w = w_solver(z0, filters_fft, parameters.theta / gamma)
        p = torch.clip(z0, 0, 1)
        z0 = z_solver(
            p, kt_fft, kernel_fft, w, filters_fft, gamma, beta[..., i]
        )
        # loss
        loss_step.append(loss(p, t, k).item())

        # stop criterion
        p_new = torch.clip(z0, 0, 1)
        if eps and torch.mean((p - p_new).abs()) <= eps:
            break

        if parameters.debug is not None:
            parameters.debug["p"] = p_new
        progress(i / parameters.max_iter)

    progress(1.0)
    return p_new


def _demo():
    from ..._demo import demo

    def demo_hqs(
        image: Tensor, kernel: Tensor, progress: Callable[[float], None]
    ) -> Tensor:
        return hqs_pc(image, kernel, HQS_PC_Parameters(progress=progress))

    demo("HQS PC", demo_hqs)


if __name__ == "__main__":
    _demo()
