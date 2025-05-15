import torch
from torch import Tensor
from . import PrecompensationKERUNC
from torchvision.transforms.functional import rgb_to_grayscale


def _demo():
    from ...._demo import demo
    from typing import Callable

    def demo_dwdn(
        image: Tensor, psf: Tensor, progress: Callable[[float], None]
    ) -> Tensor:
        model = PrecompensationKERUNC() # .from_path(path="hf://RVI/dwdn.pt")

        with torch.inference_mode():
            image_gray = rgb_to_grayscale(image) 
            inputs = model.preprocess(image_gray, psf.to(torch.float32))
            progress(0.1)
            (precompensation,) = model(inputs)
            progress(1.0)
            return precompensation

    demo("DWDN", demo_dwdn, mono=True, num_output_channels=3)


if __name__ == "__main__":
    _demo()
