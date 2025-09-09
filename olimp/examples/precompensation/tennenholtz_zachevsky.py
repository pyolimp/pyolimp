from __future__ import annotations

from typing import Callable
from torch import Tensor
import warnings

from olimp.precompensation._demo_cvd import demo as demo_cvd
from olimp.precompensation.optimization.tennenholtz_zachevsky import (
    tennenholtz_zachevsky,
    TennenholtzZachevskyParameters,
)
from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion

if __name__ == "__main__":

    def demo_tennenholtz_zachevsky(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> tuple[Tensor]:
        parameters = TennenholtzZachevskyParameters(progress=progress)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                tennenholtz_zachevsky(image[0], distortion, parameters)[None],
            )

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo_cvd(
        "Tennenholtz-Zachevsky",
        demo_tennenholtz_zachevsky,
        distortion=distortion,
    )
