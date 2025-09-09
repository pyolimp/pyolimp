from __future__ import annotations

import torch
from torch import Tensor
from matplotlib import pylab as plt

from olimp.simulate.psf_sca import PSFSCA
from olimp.simulate.refraction_distortion import RefractionDistortion
from olimp.simulate._demo_distortion import demo


def show_one(image: Tensor, title: str) -> None:
    if image.isnan().any():
        raise ValueError("has nan")
    fig, ax1 = plt.subplots(dpi=72, figsize=(6, 4.5), ncols=1, nrows=1)
    plt.title(title)
    ax1.imshow(image)
    plt.show()


if __name__ == "__main__":

    def demo_simulate():
        psf1 = PSFSCA(512, 512)()
        yield RefractionDistortion()(torch.fft.fftshift(psf1)), "psf1"

        psf2 = PSFSCA(512, 512)(sphere_dpt=-3, cylinder_dpt=-2)
        yield RefractionDistortion()(torch.fft.fftshift(psf2)), "psf2"

    demo("RefractionDistortion", demo_simulate, on="horse", size=(512, 512))
