from __future__ import annotations
import torch
from torch import nn
from torch import Tensor

# import torchvision
from olimp.processing import fft_conv
from .download_path import download_path, PyOlimpHF


class VAE(nn.Module):
    """
    .. image:: ../../../../_static/vae.svg
       :class: full-width
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Assuming input image size is 512x512
        self.fc_mu = nn.Linear(1024 * 8 * 8, 128)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, 128)

        # Decoder
        self.decoder_input = nn.Linear(128, 1024 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 512, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                512, 256, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, 3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    @classmethod
    def from_path(cls, path: PyOlimpHF):
        model = cls()
        path = download_path(path)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def reparameterize(self, mu: nn.Linear, logvar: nn.Linear):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        z = self.reparameterize(mu, logvar)

        decoded = self.decoder_input(z)
        decoded = decoded.view(-1, 1024, 8, 8)
        decoded = self.decoder(decoded)
        return decoded, mu, logvar

    def preprocess(self, image: Tensor, psf: Tensor) -> Tensor:
        # img_gray = image.to(torch.float32)[None, ...]
        # img_gray = torchvision.transforms.Resize((512, 512))(img_gray)
        img_blur = fft_conv(image, psf)

        return torch.cat(
            [
                image,
                img_blur,
                psf,
            ],
            dim=1,
        )

    def postprocess(self, tensors: tuple[Tensor]) -> tuple[Tensor]:
        return tensors

    def arguments(self, *args):
        return {}


def _demo():
    from ..._demo import demo
    from typing import Callable

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


if __name__ == "__main__":
    _demo()
