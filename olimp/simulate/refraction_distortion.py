from __future__ import annotations
from torch import Tensor
from olimp.simulate import ApplyDistortion, Distortion
from olimp.processing import fft_conv


class RefractionDistortion(Distortion):
    """
    .. image:: ../_static/refraction_distortion.svg
       :class: full-width

    .. important::
       psf must be shifted with `torch.fft.fftshift` and its sum
       must be equal to 1.
    """

    @staticmethod
    def __call__(psf: Tensor) -> ApplyDistortion:
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
