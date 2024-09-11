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
            inputs = model.preprocess(image, psf)

            noise_level = 0
            sigma = torch.tensor(noise_level).float().view([1, 1, 1, 1])
            sigma = sigma.repeat([inputs.shape[0], 1, 1, 1])
            scale_factor = 1

            progress(0.1)
            precompensation = model(
                inputs, torch.fft.fftshift(psf), scale_factor, sigma
            )
            progress(1.0)
            return precompensation[0, 0]

    demo("USRNET", demo_usrnet, mono=True)


if __name__ == "__main__":
    _demo()
