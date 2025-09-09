from __future__ import annotations

from torch import Tensor
from matplotlib import pylab as plt

from olimp.precompensation.nn.dataset.psf_sca import PSFSCADataset
from olimp.simulate._demo_distortion import demo


def show_one(image: Tensor, title: str) -> None:
    if image.isnan().any():
        raise ValueError("has nan")
    fig, ax1 = plt.subplots(dpi=72, figsize=(6, 4.5), ncols=1, nrows=1)
    plt.title(title)
    ax1.imshow(image)
    plt.show()


if __name__ == "__main__":

    dataset = PSFSCADataset(
        width=512,
        height=512,
        sphere_dpt={
            "name": "uniform",
            "a": -4.0,
            "b": -2.0,
        },  # uniform в диапазоне [-2, 0]
        cylinder_dpt={
            "name": "uniform",
            "a": -4.0,
            "b": -2.0,
        },  # то, что у вас уже было
        angle_deg={
            "name": "uniform",
            "a": 0.0,
            "b": 180.0,
        },  # то, что у вас уже было
        pupil_diameter_mm={
            "name": "uniform",
            "a": 3.0,
            "b": 5.0,
        },  # uniform, например, от 3 до 5 мм
        am2px=0.001,
        seed=42,
        size=100,
    )

    def demo_simulate():
        apply_fn = dataset.apply()
        funcs = []

        for i in range(3):
            psf = dataset[i]
            show_one(psf[0], title=f"Gaussian PSF #{i}")

            # создаём функцию с захваченным индексом
            funcs.append((lambda image, i=i: list(apply_fn(image))[i], f"{i}"))

        return funcs

    demo("RefractionDistortion", demo_simulate, on="horse", size=(512, 512))
