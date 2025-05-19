from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from ._base import ReducibleLoss, Reduction
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
    rng = torch.Generator(device=img1.device).manual_seed(seed)

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
    image: Tensor, cy: Tensor, cx: Tensor, ny: Tensor, nx: Tensor
) -> Tensor:
    return torch.norm(image[:, cy, cx] - image[:, ny, nx], dim=0)


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

    valid_mask = (
        (0 <= neighbors[:, :, 0, :])
        & (neighbors[:, :, 0, :] < height)
        & (0 <= neighbors[:, :, 1, :])
        & (neighbors[:, :, 1, :] < width)
    )

    ny = neighbors[:, :, 0, :].clamp(0, height - 1)
    nx = neighbors[:, :, 1, :].clamp(0, width - 1)

    y_indices = torch.arange(0, dst_height) * step
    x_indices = torch.arange(0, dst_width) * step
    grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing="ij")
    cy = grid_y.unsqueeze(-1).expand(-1, -1, n_pixel_neighbors)
    cx = grid_x.unsqueeze(-1).expand(-1, -1, n_pixel_neighbors)

    image1_contrast = pixel_contrasts(lab1, cy, cx, ny, nx)
    image2_contrast = pixel_contrasts(lab2, cy, cx, ny, nx)
    contrast_diff = (image1_contrast - image2_contrast) / 1.6
    mean = (contrast_diff**2 * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)
    return torch.sqrt(mean)


class RMS(ReducibleLoss):
    """
    Root-mean-square metric based on
    https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_PCA/Machado_Oliveira_EuroVis2010.pdf
    """

    _color_space: Literal["lab", "prolab"]

    def __init__(
        self,
        color_space: Literal["lab", "prolab"],
        n_pixel_neighbors: int = 1000,
        step: int = 10,
        sigma_rate: float = 0.25,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self._color_space = color_space
        self._n_pixel_neighbors = n_pixel_neighbors
        self._step = step
        self._sigma_rate = sigma_rate

    def _loss(self, img1: Tensor, img2: Tensor):
        return torch.mean(
            RMS_map(
                img1,
                img2,
                self._color_space,
                self._n_pixel_neighbors,
                self._step,
                self._sigma_rate,
            )
        )
