from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from typing import Tuple
import math

# import torchvision
from olimp.processing import fft_conv
from .download_path import download_path, PyOlimpHF


class VAE(nn.Module):
    """
    .. image:: ../../../../_static/vae.svg
       :class: full-width
    """

    def __init__(
        self,
        input_channel: int = 3,
        output_channel: int = 1,
        image_size: Tuple[int, int] = (512, 512),
        latent_dimension: Tuple[int, int] = 128,
    ):
        super().__init__()

        # Image cast size
        self.image_size = image_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, stride=2, padding=1),
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

        # Assuming input image size is image_size
        w, h = self.image_size[0] / 64, self.image_size[1] / 64
        self.latent_shape = (math.ceil(w), math.ceil(h))
        latent_size_multiplication = (
            self.latent_shape[0] * self.latent_shape[1]
        )

        self.fc_mu = nn.Linear(
            1024 * latent_size_multiplication,
            latent_dimension,
        )
        self.fc_logvar = nn.Linear(
            1024 * latent_size_multiplication,
            latent_dimension,
        )

        # Decoder
        self.decoder_input = nn.Linear(
            latent_dimension,
            1024 * latent_size_multiplication,
        )

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
                32, output_channel, 3, stride=2, padding=1, output_padding=1
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
        decoded = decoded.view(
            -1, 1024, self.latent_shape[0], self.latent_shape[1]
        )
        decoded = self.decoder(decoded)
        decoded = torch.nn.functional.interpolate(
            decoded, size=self.image_size, mode="bilinear", align_corners=False
        )
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
