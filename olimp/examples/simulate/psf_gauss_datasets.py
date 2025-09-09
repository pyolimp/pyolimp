from __future__ import annotations

from torch import Tensor
from matplotlib import pylab as plt

from olimp.precompensation.nn.dataset.psf_gauss import PsfGaussDataset
from olimp.simulate._demo_distortion import demo


def show_one(image: Tensor, title: str) -> None:
    if image.isnan().any():
        raise ValueError("has nan")
    fig, ax1 = plt.subplots(dpi=72, figsize=(6, 4.5), ncols=1, nrows=1)
    plt.title(title)
    ax1.imshow(image)
    plt.show()


if __name__ == "__main__":

    dataset = PsfGaussDataset(
        width=512,
        height=512,
        center_x={
            "name": "uniform",
            "a": 255.0,
            "b": 257.0,
        },  # центр по X в пикселях от 0 до 63
        center_y={
            "name": "uniform",
            "a": 255.0,
            "b": 257.0,
        },  # центр по Y в пикселях от 0 до 63
        theta={
            "name": "uniform",
            "a": 0.0,
            "b": 180.0,
        },  # угол поворота в градусах
        sigma_x={
            "name": "uniform",
            "a": 5.0,
            "b": 25.0,
        },  # sigma по X в пикселях
        sigma_y={
            "name": "uniform",
            "a": 5.0,
            "b": 25.0,
        },  # sigma по Y в пикселях
        seed=99,
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
