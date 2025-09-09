from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.nn.models.vae import VAE

if __name__ == "__main__":

    def demo_vae(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = VAE.from_path("hf://RVI/vae.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            precompensation, _mu, _logvar = model(inputs)
            progress(1.0)
            return precompensation

    demo("VAE", demo_vae, mono=True)
