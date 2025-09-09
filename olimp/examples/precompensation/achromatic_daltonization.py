from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo_cvd import demo as demo_cvd
from olimp.precompensation.optimization.achromatic_daltonization import (
    achromatic_daltonization,
    ColorBlindnessDistortion,
    ADParameters,
    M1Loss,
)
import warnings


def demo_achromatic_daltonization(
    image: Tensor,
    distortion: ColorBlindnessDistortion,
    progress: Callable[[float], None],
) -> tuple[Tensor]:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            achromatic_daltonization(
                image,
                distortion,
                ADParameters(progress=progress, loss_func=M1Loss()),
            ),
        )


distortion = ColorBlindnessDistortion.from_type("protan")
demo_cvd(
    "Achromatic Daltonization",
    demo_achromatic_daltonization,
    distortion=distortion,
)
