from __future__ import annotations

from typing import Callable
from torch import Tensor

from olimp.processing import scale_value, fft_conv
from olimp.precompensation._demo import demo
from olimp.precompensation.analytics.feng_xu import feng_xu

if __name__ == "__main__":

    def demo_huang(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        ret = feng_xu(image, psf, lambda_val=2)
        progress(0.8)
        ret = fft_conv(scale_value(ret), psf)
        progress(1.0)
        return ret

    demo("Feng Xu", demo_huang, mono=True)
