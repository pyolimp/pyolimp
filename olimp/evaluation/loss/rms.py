from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from ..cs import D65 as D65_sRGB
from ..cs.srgb import sRGB
from ..cs.cielab import CIELAB
from ..cs.prolab import ProLab


def generate_random_neighbors(
    img1: Tensor,
    img2: Tensor,
    n_pixel_neighbors: int = 1000,
    step: int = 10,
    sigma_rate: float = 0.25,
) -> Tensor:
    _channels, height, width = img1.shape
    dst_height, dst_width = height // step, width // step

    sigma = torch.tensor([height * sigma_rate, width * sigma_rate])

    # Create a grid of indices using meshgrid
    y_indices = torch.arange(0, dst_height) * step
    x_indices = torch.arange(0, dst_width) * step
    indices = torch.stack(
        torch.meshgrid(y_indices, x_indices, indexing="ij"), dim=-1
    )

    seed = hash(torch.mean(img1 + img2).item())
    rng = torch.Generator().manual_seed(seed)

    neighbors = torch.empty(
        (dst_height, dst_width, 2, n_pixel_neighbors), dtype=torch.float32
    )

    torch.normal(
        indices.unsqueeze(-1).repeat(1, 1, 1, n_pixel_neighbors),
        sigma.unsqueeze(-1).repeat(1, 1, 1, n_pixel_neighbors),
        generator=rng,
        out=neighbors,
    )
    return neighbors.round().clamp(min=0).long()


def projective_transformation(points: Tensor, proj_matrix: Tensor) -> Tensor:
    cartesian_index = proj_matrix.shape[0] - 1
    points_homog = torch.cat(
        (points, torch.ones(points.shape[0], 1, device=points.device)), dim=1
    )
    proj_points_homog = points_homog @ proj_matrix.T
    projection = proj_points_homog / proj_points_homog[:, cartesian_index:]
    projection = projection[:, :cartesian_index]
    return projection


def srgb2prolab(srgb: Tensor) -> Tensor:
    return ProLab(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def srgb2lab(srgb: Tensor) -> Tensor:
    return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def pixel_contrasts(
    image: Tensor, pixel: tuple[int, int], neighbors: tuple[Tensor, ...]
) -> Tensor:
    pixel_value = image[:, pixel[0], pixel[1]]
    neighbor_values = image[:, neighbors[0], neighbors[1]]
    contrasts = torch.norm(pixel_value[:, None] - neighbor_values, dim=0)
    return contrasts


def RMS_map(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
    n_pixel_neighbors: int = 1000,
    step: int = 10,
    sigma_rate: float = 0.25,
) -> Tensor:
    _channels, height, width = img1.shape
    dst_height, dst_width = height // step, width // step

    neighbors = generate_random_neighbors(
        img1, img2, n_pixel_neighbors, step, sigma_rate
    )

    # calculate rms
    if color_space == "lab":
        lab1 = srgb2lab(img1)
        lab2 = srgb2lab(img2)

    elif color_space == "prolab":
        lab1 = srgb2prolab(img1)
        lab2 = srgb2prolab(img2)

    rms = torch.zeros((dst_height, dst_width))

    for i in torch.arange(dst_height):
        for j in torch.arange(dst_width):
            pixel_neighbors = neighbors[i, j, :, :]
            filtered_neighbors = pixel_neighbors[
                :,
                torch.all(
                    (pixel_neighbors >= 0)
                    & (pixel_neighbors < torch.tensor([[height], [width]])),
                    dim=0,
                ),
            ]
            img1_contrasts = pixel_contrasts(
                lab1, (int(i * step), int(j * step)), tuple(filtered_neighbors)
            )
            img2_contrasts = pixel_contrasts(
                lab2, (int(i * step), int(j * step)), tuple(filtered_neighbors)
            )
            normalized_contrast_diffs = (img1_contrasts - img2_contrasts) / 1.6
            rms[i, j] = torch.sqrt(torch.mean(normalized_contrast_diffs**2))

    return rms


class RMS(Module):
    _color_space: Literal["lab", "prolab"]

    def __init__(
        self,
        color_space: Literal["lab", "prolab"],
        n_pixel_neighbors: int = 1000,
        step: int = 10,
        sigma_rate: float = 0.25,
    ):
        super().__init__()
        self._color_space = color_space
        self._n_pixel_neighbors = n_pixel_neighbors
        self._step = step
        self._sigma_rate = sigma_rate

    def forward(self, img1: Tensor, img2: Tensor):
        assert img1.ndim == 4, img1.shape
        assert img2.ndim == 4, img2.shape

        assert img1.shape[1] == 3
        assert img2.shape[1] == 3

        rms_maps = torch.empty((img1.shape[0]))
        for idx in range(img1.shape[0]):
            rms_maps[idx] = torch.mean(
                RMS_map(
                    img1[idx],
                    img2[idx],
                    self._color_space,
                    self._n_pixel_neighbors,
                    self._step,
                    self._sigma_rate,
                )
            )
        return rms_maps
