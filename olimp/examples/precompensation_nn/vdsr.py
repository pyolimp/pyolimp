from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.nn.models.vdsr import VDSR

if __name__ == "__main__":

    def demo_vdsr(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = VDSR.from_path("hf://RVI/vdsr.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            (precompensation,) = model(inputs)
            progress(1.0)
            return precompensation

    demo("VDSR", demo_vdsr, mono=True)
