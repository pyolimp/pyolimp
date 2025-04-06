from __future__ import annotations
from typing import TypeAlias
import torch
import torch.nn as nn
from torch import Tensor
from olimp.processing import fft_conv
from .download_path import download_path, PyOlimpHF

# import torch.nn.functional as F


Input: TypeAlias = tuple[Tensor, int]


class CVAE(nn.Module):
    """
    .. image:: ../../../../_static/cvae.svg
       :class: full-width
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()

        self.num_classes = num_classes

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

        self.fc_mu = nn.Linear(1024 * 8 * 8 + num_classes, 128)
        self.fc_logvar = nn.Linear(1024 * 8 * 8 + num_classes, 128)

        # Decoder
        self.decoder_input = nn.Linear(128 + num_classes, 1024 * 8 * 8)

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs: Input):
        x, y = inputs
        # One-hot encode y
        # y = F.one_hot(y, num_classes=self.num_classes).float()
        y = y.view(y.size(0), -1)  # Flatten condition

        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)

        # Concatenate encoded features with y
        encoded = torch.cat([encoded, y], dim=1)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Concatenate z with y
        z = torch.cat([z, y], dim=1)

        # Decode
        decoded = self.decoder_input(z)
        decoded = decoded.view(-1, 1024, 8, 8)
        decoded = self.decoder(decoded)
        return decoded, mu, logvar

    @classmethod
    def from_path(cls, path: PyOlimpHF):
        model = cls()
        path = download_path(path)
        state_dict = torch.load(
            path, map_location=torch.get_default_device(), weights_only=True
        )
        model.load_state_dict(state_dict)
        return model

    def preprocess(self, image: Tensor, psf: Tensor) -> Tensor:
        image_low_contrast = (image * (0.7 - 0.3)) + 0.3
        retinal_original = fft_conv(image_low_contrast, psf)
        x = torch.cat([image_low_contrast, retinal_original, psf], dim=1)
        y = torch.ones(x.size(0), 1)
        return x, y

    def postprocess(self, tensors: tuple[Tensor]) -> tuple[Tensor]:
        return tensors

    def arguments(self, *args):
        return {}


def _demo():
    from ..._demo import demo
    from typing import Callable

    def demo_cvae(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = CVAE.from_path("hf://RVI/cvae.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            (precompensation, mu, logvar) = model(inputs)
            progress(1.0)
            return precompensation

    demo("CVAE", demo_cvae, mono=True)


if __name__ == "__main__":
    _demo()
