from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.processing import scale_value
from olimp.precompensation._demo import demo
from olimp.precompensation.basic.huang import huang

if __name__ == "__main__":

    def demo_huang(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        ret = huang(image, psf)
        progress(1.0)
        return scale_value(ret, min_val=0, max_val=1.0)

    demo("Huang", demo_huang, mono=False)
