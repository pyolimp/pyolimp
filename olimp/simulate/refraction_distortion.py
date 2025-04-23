from __future__ import annotations
import torch
from torch import Tensor
from olimp.simulate import ApplyDistortion, Distortion
from olimp.processing import fft_conv
from olimp.precompensation.nn.dataset.psf_geometric import (
    geometric_psf_generator,
)


class RefractionDistortion(Distortion):
    """
    .. image:: ../_static/refraction_distortion.svg
       :class: full-width

    .. important::
       psf must be shifted with `torch.fft.fftshift` and its sum
       must be equal to 1.
    """

    def __init__(self, psf: Tensor | None = None) -> None:
        self.psf = psf

    @classmethod
    def from_sca_params(
        cls,
        width: int,
        height: int,
        sphere_dpt: float,
        cylinder_dpt: float,
        angle_rad: float,
        pupil_diameter_mm: float,
        am2px: float = 0.001,
    ):
        x = torch.arange(width, dtype=torch.float32)
        y = torch.arange(height, dtype=torch.float32)
        x = x - width * 0.5
        y = y - height * 0.5
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        return geometric_psf_generator(
            sphere_dpt,
            cylinder_dpt,
            angle_rad,
            pupil_diameter_mm,
            x,
            y,
            am2px,
        )

    def __call__(self, psf: Tensor | None = None) -> ApplyDistortion:
        if psf is None:
            assert self.psf is not None
            psf = self.psf
        return lambda image: fft_conv(image, psf)


def _demo():
    from ._demo_distortion import demo

    def demo_simulate():
        from pathlib import Path
        import torch
        import numpy as np

        root = Path(__file__).parents[2]
        for suffix in ("2", ""):
            psf_info = np.load(root / f"tests/test_data/psf{suffix}.npz")
            psf = torch.fft.fftshift(
                torch.tensor(psf_info["psf"]).to(torch.float32)
            )

            yield RefractionDistortion()(psf), f"psf{suffix}.npz"

    demo(
        "RefractionDistortion", demo_simulate, on="horse.jpg", size=(512, 512)
    )


if __name__ == "__main__":
    _demo()
