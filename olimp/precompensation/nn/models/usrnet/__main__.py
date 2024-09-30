import torch
from . import PrecompensationUSRNet


def _demo():
    from ...._demo import demo
    from typing import Callable

    def demo_usrnet(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        model = PrecompensationUSRNet.from_path(
            path="./olimp/weights/usrnet.pth"
        )
        with torch.no_grad():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(
                image, psf, scale_factor=1, noise_level=0
            )

            progress(0.1)
            precompensation = model(inputs)
            progress(1.0)
            return precompensation[0, 0]

    demo("USRNET", demo_usrnet, mono=True, num_output_channels=3)


if __name__ == "__main__":
    _demo()
