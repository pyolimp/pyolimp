from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.optimization.bregman_jumbo import (
    bregman_jumbo,
    BregmanJumboParameters,
)


if __name__ == "__main__":

    def demo_bregman_jumbo(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return bregman_jumbo(
            image, psf, BregmanJumboParameters(progress=progress)
        )

    demo("Bregman Jumbo", demo_bregman_jumbo)
