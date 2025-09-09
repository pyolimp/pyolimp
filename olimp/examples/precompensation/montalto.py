from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.optimization.montalto import (
    montalto as montalto,
    MontaltoParameters,
)


if __name__ == "__main__":

    def demo_montalto(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return montalto(image, psf, MontaltoParameters(progress=progress))

    demo("Montalto", demo_montalto)
