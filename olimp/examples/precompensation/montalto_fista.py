from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.optimization.montalto_fista import (
    montalto as montalto_fista,
    MontaltoParameters as FistaMontaltoParameters,
)

if __name__ == "__main__":

    def demo_montalto(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return montalto_fista(
            image, psf, FistaMontaltoParameters(progress=progress)
        )

    demo("Montalto (FISTA)", demo_montalto, mono=False)
