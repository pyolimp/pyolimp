from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.nn.models.unet_efficient_b0 import (
    PrecompensationUNETB0,
)

if __name__ == "__main__":

    def demo_unet(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = PrecompensationUNETB0.from_path(
            "hf://RVI/unet-efficientnet-b0.pth"
        )
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            (precompensation,) = model(inputs)
            progress(1.0)
            return precompensation

    demo("UNET", demo_unet, mono=True)
