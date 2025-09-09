from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.optimization.global_tone_mapping import (
    precompensation_global_tone_mapping,
    GTMParameters,
)


if __name__ == "__main__":

    def demo_global_tone_mapping(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        return precompensation_global_tone_mapping(
            image, psf, GTMParameters(progress=progress, lr=0.05)
        )

    demo("Global Tone Mapping", demo_global_tone_mapping, mono=False)
