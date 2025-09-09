from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.optimization.hqs import hqs, HQSParameters

if __name__ == "__main__":

    def demo_hqs(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return hqs(image, psf, HQSParameters(progress=progress))

    demo("Half-Quadratic", demo_hqs)
