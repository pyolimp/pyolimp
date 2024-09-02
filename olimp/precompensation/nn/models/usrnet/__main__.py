import torch
from . import PrecompensationUSRNet


def _demo():
    from ...._demo import demo
    from typing import Callable
    import torchvision

    def demo_usrnet(
        image: torch.Tensor,
        psf: torch.Tensor,
        progress: Callable[[float], None],
    ) -> torch.Tensor:
        model = PrecompensationUSRNet.from_path(
            path="./olimp/weights/usrnet.pth"
        )
        with torch.no_grad():
            psf = psf.to(torch.float32).unsqueeze(0).unsqueeze(0)
            batch_size = 1
            noise_level = 0
            sigma = torch.tensor(noise_level).float().view([1, 1, 1, 1])
            sigma = sigma.repeat([batch_size, 1, 1, 1])
            scale_factor = 1
            resize_transform = torchvision.transforms.Resize((512, 512))

            image = resize_transform(image)
            img_gray = image.to(torch.float32)[None, None, ...]
            img_gray = torchvision.transforms.Resize((512, 512))(img_gray)
            # lower image contrast to make this demo look good
            img_gray = (img_gray * (0.7 - 0.3)) + 0.3

            inputs = torch.cat(
                [img_gray, img_gray, img_gray], dim=1
            )  # B, 3, 512, 512

            progress(0.1)
            precompensation = model(
                inputs, torch.fft.fftshift(psf), scale_factor, sigma
            )
            progress(1.0)
            return precompensation[0, 0]

    demo("USRNET", demo_usrnet, mono=True)


if __name__ == "__main__":
    _demo()
