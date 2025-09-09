from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.nn.models.dwdn import PrecompensationDWDN

if __name__ == "__main__":

    def demo_dwdn(
        image: Tensor, psf: Tensor, progress: Callable[[float], None]
    ) -> Tensor:
        model = PrecompensationDWDN.from_path(path="hf://RVI/dwdn.pt")

        with torch.inference_mode():
            inputs = model.preprocess(image, psf.to(torch.float32))
            progress(0.1)
            (precompensation,) = model(inputs, **model.arguments(inputs, psf))
            progress(1.0)
            return precompensation

    demo("DWDN", demo_dwdn, mono=True, num_output_channels=3)
