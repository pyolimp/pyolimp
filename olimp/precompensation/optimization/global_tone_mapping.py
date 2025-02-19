from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
from torch import Tensor

import torch

from olimp.processing import fft_conv
from olimp.precompensation.basic.huang import huang
from olimp.evaluation.loss.piq import MultiScaleSSIMLoss


class DebugInfo(TypedDict):
    loss_step: list[float]
    precomp: torch.Tensor


class GTMParameters(NamedTuple):
    x1: float = -1.1
    x2: float = 1.1
    y1: float = 0.1
    y2: float = 0.9
    loss_func: Callable[[Tensor, Tensor], Tensor] | None = None
    optimizer_tonemapping: type[torch.optim.Optimizer] | None = None
    k: float = 0.01
    lr: float = 0.01
    iterations: int = 500
    gap: float = 0.001
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None  # Pass dictionary for debugging
    history_loss: list[float] | None = None  # []


def apply_global_tone_mapping(
    precomp: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    eps: float = 1e-1,
) -> Tensor:
    # Identify ranges for precomp
    below_x1 = torch.lt(precomp, x1)
    above_x2 = torch.gt(precomp, x2)
    between_x1_x2 = ~(below_x1 | above_x2)

    # Compute tone mapping for different ranges with stability improvements
    mapped_below_x1 = y1 * torch.exp(1 - (y2 / (y1 + eps)) / ((x2 - x1) + eps))
    mapped_above_x2 = 1 - (
        (1 - y2)
        * torch.exp(
            ((precomp - x2) / ((x1 - x2) + eps))
            * ((y2 - y1) / ((1 - y2) + eps))
        )
    )
    mapped_between_x1_x2 = y1 + (
        (y2 - y1) * ((precomp - x1) / ((x2 - x1) + eps))
    )

    # Combine ranges into a single normalized output
    normalized_precomp = torch.where(
        below_x1,
        mapped_below_x1,
        torch.where(above_x2, mapped_above_x2, mapped_between_x1_x2),
    )

    # Ensure consistency within the 'in-between' range
    normalized_precomp = torch.where(
        between_x1_x2, mapped_between_x1_x2, normalized_precomp
    )

    # Clamp final values to prevent numerical instability
    normalized_precomp = torch.clamp(normalized_precomp, 0.0, 1.0)

    return normalized_precomp


def precompensation_global_tone_mapping(
    img: Tensor,
    psf: Tensor,
    params: GTMParameters,
) -> Tensor:
    """
    .. image:: ../../_static/global_tone_mapping.svg
       :class: full-width
    """
    x1 = torch.tensor([params.x1], requires_grad=True)
    x2 = torch.tensor([params.x2], requires_grad=True)
    y1 = torch.tensor([params.y1], requires_grad=True)
    y2 = torch.tensor([params.y2], requires_grad=True)
    history_loss = [] if params.history_loss is None else params.history_loss
    loss_func = (
        MultiScaleSSIMLoss() if params.loss_func is None else params.loss_func
    )
    optimizer_tonemapping = (
        torch.optim.Adam
        if params.optimizer_tonemapping is None
        else params.optimizer_tonemapping
    )

    optimizer = optimizer_tonemapping([x1, x2, y1, y2], lr=params.lr)
    precomp = huang(img, psf, k=params.k)

    for i in range(params.iterations):
        optimizer.zero_grad()

        x1.data.clamp_(max=x2.item() - 0.01)  # trick torch with .data
        x2.data.clamp_(min=x1.item() + 0.01)
        y1.data.clamp_(min=0.01, max=0.99)
        y2.data.clamp_(min=0.01, max=0.99)

        precomp_normalized = apply_global_tone_mapping(precomp, x1, x2, y1, y2)

        precomp_normaliz_retinal = fft_conv(precomp_normalized, psf)

        loss = loss_func(precomp_normaliz_retinal, img)

        loss.backward()
        optimizer.step()

        if params.debug is not None:
            params.debug["loss_step"].append(loss.item())

        if params.progress is not None:
            params.progress(i / params.iterations)

        history_loss.append(loss.item())
        if len(history_loss) > 50:
            max_change = max(history_loss) - min(history_loss)
            if max_change < params.gap:
                # print(
                #     f"Optimization stopped at iteration {i} due to low average loss change."
                # )
                break
            history_loss.pop(0)

    # Return the final optimized precompensation
    return precomp_normalized


def _demo():
    from .._demo import demo

    def demo_global_tone_mapping(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        return precompensation_global_tone_mapping(
            image, psf, GTMParameters(progress=progress, lr=0.05)
        )

    demo("Global Tone Mapping", demo_global_tone_mapping, mono=False)


if __name__ == "__main__":
    _demo()
