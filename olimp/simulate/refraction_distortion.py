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
        import torch
        from olimp.demo_data import psf as demo_psf, psf2 as demo_psf2

        for psf_gen in demo_psf, demo_psf2:
            psf_info = psf_gen()
            psf = torch.fft.fftshift(
                torch.tensor(psf_info["psf"]).to(torch.float32)
            )

            yield RefractionDistortion()(psf), f"{psf_gen.__name__}"

    demo("RefractionDistortion", demo_simulate, on="horse", size=(512, 512))


if __name__ == "__main__":
    _demo()
