from __future__ import annotations


from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo import demo
from olimp.precompensation.nn.models.usrnet import PrecompensationUSRNet

if __name__ == "__main__":

    def demo_usrnet(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = PrecompensationUSRNet.from_path(path="hf://RVI/usrnet.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(
                image, psf, scale_factor=1, noise_level=0
            )

            progress(0.1)
            (precompensation,) = model(inputs)
            progress(1.0)
            return precompensation

    demo("USRNET", demo_usrnet, mono=True, num_output_channels=3)
