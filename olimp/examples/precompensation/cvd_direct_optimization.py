from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo_cvd import demo as demo_cvd
from olimp.precompensation.optimization.cvd_direct_optimization import (
    cvd_direct_optimization,
    CVDParameters,
)
from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion


if __name__ == "__main__":

    def demo_cvd_direct_optimization(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> Tensor:
        from olimp.evaluation.loss.rms import RMS

        return (
            cvd_direct_optimization(
                image,
                distortion,
                CVDParameters(progress=progress, loss_func=RMS("lab")),
            ),
        )

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo_cvd(
        "CVD DIRECT OPTIMIZATION",
        demo_cvd_direct_optimization,
        distortion=distortion,
    )
