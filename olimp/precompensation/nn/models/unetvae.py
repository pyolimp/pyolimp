from __future__ import annotations
import torch
from torch import nn
from torch import Tensor

# import torchvision
from olimp.processing import fft_conv
from .download_path import download_path, PyOlimpHF


class UNETVAE(nn.Module):
    """
    .. image:: ../../../../_static/unetvae.svg
       :class: full-width
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(1024 * 8 * 8, 128)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, 128)
        self.decoder_input = nn.Linear(128, 1024 * 8 * 8)

        # Decoder
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 512, 3, stride=2, padding=1, output_padding=1
            ),  # 8 -> 16
            nn.ReLU(),
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, 256, 3, stride=2, padding=1, output_padding=1
            ),  # 16 -> 32
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 128, 3, stride=2, padding=1, output_padding=1
            ),  # 32 -> 64
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, 3, stride=2, padding=1, output_padding=1
            ),  # 64 -> 128
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, 3, stride=2, padding=1, output_padding=1
            ),  # 128 -> 256
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 3, 3, stride=2, padding=1, output_padding=1
            ),  # 256 -> 512
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
        # Encoder with skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)

        # Flatten for latent space
        encoded = e6.view(e6.size(0), -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        # Decode latent vector
        decoded_input = self.decoder_input(z).view(-1, 1024, 8, 8)

        # Decoder with skip connections
        d6 = self.decoder6(decoded_input)
        d5 = self.decoder5(torch.cat([d6, e5], dim=1))
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))

        return d1, mu, logvar

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

    def demo_unetvae(
        image: Tensor,
        psf: Tensor,
        progress: Callable[[float], None],
    ) -> Tensor:
        model = UNETVAE.from_path("hf://RVI/unetvae.pth")
        with torch.inference_mode():
            psf = psf.to(torch.float32)
            inputs = model.preprocess(image, psf)
            progress(0.1)
            precompensation, _mu, _logvar = model(inputs)
            progress(1.0)
            return precompensation

    demo("UNETVAE", demo_unetvae, mono=True)


if __name__ == "__main__":
    _demo()
