from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.optimization.ji import ji, JiParameters
from olimp.evaluation.loss.piq import MultiScaleSSIMLoss


def demo_ji(
    image: Tensor,
    psf: Tensor,
    progress: Callable[[float], None],
) -> Tensor:
    return ji(
        image,
        psf,
        JiParameters(
            progress=progress, alpha=1, loss_func=MultiScaleSSIMLoss()
        ),
    )


demo("Ji", demo_ji)
