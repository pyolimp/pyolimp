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
    x1: Tensor = torch.tensor([-1.1], requires_grad=True)
    x2: Tensor = torch.tensor([1.1], requires_grad=True)
    y1: Tensor = torch.tensor([0.1], requires_grad=True)
    y2: Tensor = torch.tensor([0.9], requires_grad=True)
    loss_func: Callable[[Tensor, Tensor], Tensor] = MultiScaleSSIMLoss()
    optimizer_tonemapping: Callable = torch.optim.Adam
    k: float = 0.01
    lr: float = 0.01
    iterations: int = 500
    gap: float = 0.001
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None  # Pass dictionary for debugging
    history_loss: list[float] = []


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
    between_x1_x2 = torch.logical_and(
        torch.ge(precomp, x1), torch.le(precomp, x2)
    )

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
    optimizer = params.optimizer_tonemapping(
        [params.x1, params.x2, params.y1, params.y2], lr=params.lr
    )

    precomp = huang(img, psf, k=params.k)

    for i in range(params.iterations):
        optimizer.zero_grad()

        params.x1.data.clamp_(max=params.x2.item() - 0.01)
        params.x2.data.clamp_(min=params.x1.item() + 0.01)
        params.y1.data.clamp_(min=0.01, max=0.99)
        params.y2.data.clamp_(min=0.01, max=0.99)

        precomp_normaliz = apply_global_tone_mapping(
            precomp, params.x1, params.x2, params.y1, params.y2
        )

        precomp_normaliz_retinal = fft_conv(precomp_normaliz, psf)

        loss = params.loss_func(precomp_normaliz_retinal, img)

        loss.backward()
        optimizer.step()

        if params.debug is not None:
            params.debug["loss_step"].append(loss.item())

        if params.progress is not None:
            params.progress(i / params.iterations)

        params.history_loss.append(loss.item())
        if len(params.history_loss) > 10:  # Сравниваем последние 10 итераций
            params.history_loss.pop(0)
            avg_change = sum(
                abs(params.history_loss[i] - params.history_loss[i - 1])
                for i in range(1, len(params.history_loss))
            ) / (len(params.history_loss) - 1)
            if avg_change < params.gap:
                print(
                    f"Optimization stopped at iteration {i} due to low average loss change."
                )
                break

    # Return the final optimized precompensation
    return precomp_normaliz


def _demo():
    from .._demo import demo

    def demo_global_tone_mapping(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        return precompensation_global_tone_mapping(
            image, psf, GTMParameters(progress=progress)
        )

    demo("Global tone mapping", demo_global_tone_mapping, mono=True)


if __name__ == "__main__":
    _demo()
