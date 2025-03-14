from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion


class DebugInfo(TypedDict):
    loss_step: list[float]
    precomp: Tensor


class CVDParameters(NamedTuple):
    lr: float = 1e-2
    gap: float = 0.0001
    loss_func: Callable[[Tensor, Tensor], Tensor] | None = None
    progress: Callable[[float], None] | None = None
    debug: None | DebugInfo = None  # pass dict to log


def cvd_direct_optimization(
    image: Tensor,
    distortion: ColorBlindnessDistortion,
    parameters: CVDParameters = CVDParameters(),
) -> Tensor:
    """
    .. image:: ../../_static/cvd_optimization.svg
       :class: full-width
    """

    assert image.ndim == 4, image.shape

    # Parametrs
    lr = parameters.lr

    prev_loss = torch.tensor(float("inf"))

    weight_map = torch.ones(image.shape[-2:], requires_grad=True)
    precomp_image = image.clone()

    distortion_apply = distortion()

    optimizer = torch.optim.Adam([weight_map], lr=lr)
    loss_step = []
    if parameters.debug is not None:
        parameters.debug["loss_step"] = loss_step

    for i in range(5000):
        if parameters.progress is not None:
            parameters.progress(i / 5000)
        optimizer.zero_grad()

        precomp_image = (image * weight_map).clip(0, 1)

        retinal_image = distortion_apply(precomp_image)

        loss = parameters.loss_func(retinal_image, image)

        loss_step.append(loss.item())
        loss.backward()
        optimizer.step()

        if parameters.debug is not None:
            parameters.debug["precomp"] = weight_map

        if torch.abs(prev_loss - loss).item() < parameters.gap:
            break

        prev_loss = loss
    if parameters.progress is not None:
        parameters.progress(1.0)

    return precomp_image.clip_(0, 1)


def _demo():
    from .._demo_cvd import demo

    def demo_cvd_direct_optimization(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        from olimp.evaluation.loss.rms import RMS

        return (
            cvd_direct_optimization(
                image,
                distortion,
                CVDParameters(progress=progress, loss_func=RMS("lab")),
            ),
        )

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo(
        "CVD DIRECT OPTIMIZATION",
        demo_cvd_direct_optimization,
        distortion=distortion,
    )


if __name__ == "__main__":
    _demo()
