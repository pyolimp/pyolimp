from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.nn.models.cvae import CVAE

if __name__ == "__main__":

    def demo_cvae(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = CVAE.from_path("hf://RVI/cvae.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            (precompensation, mu, logvar) = model(inputs)
            progress(1.0)
            return precompensation

    demo("CVAE", demo_cvae, mono=True)
