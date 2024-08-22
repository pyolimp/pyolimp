from typing import Literal

from piq import MultiScaleSSIMLoss
import torch
from torch import Tensor
from skimage import color
import colour


def ms_ssim(pred: Tensor, target: Tensor) -> Tensor:
    _ms_ssim = MultiScaleSSIMLoss()
    loss = _ms_ssim(pred, target)
    return loss


def L1_KLD(pred: Tensor, target: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    # L1 loss for reconstruction
    L1 = torch.nn.functional.l1_loss(pred, target, reduction="sum")
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L1 + KLD


def generate_random_neighbors(
    img1: Tensor,
    img2: Tensor,
    n_pixel_neighbors: int = 1000,
    step: int = 10,
    sigma_rate: float = 0.25,
) -> Tensor:
    height, width, _ = img1.shape
    dst_height, dst_width = height // step, width // step

    sigma = torch.tensor([height * sigma_rate, width * sigma_rate])

    # Create a grid of indices using meshgrid
    y_indices = torch.arange(0, dst_height) * step
    x_indices = torch.arange(0, dst_width) * step
    indices = torch.stack(torch.meshgrid(y_indices, x_indices), dim=-1)

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


class ProLab:
    def __init__(
        self, illuminant_XYZ: Tensor = torch.tensor([0.95047, 1.0, 1.08883])
    ):
        self.illuminant_XYZ = illuminant_XYZ
        self.Q = torch.tensor(
            [
                [75.5644333, 486.62630402, 167.39926268, 0.0],
                [617.72787198, -595.4607401, -22.26712291, 0.0],
                [48.3448951, 194.93477285, -243.27966363, 0.0],
                [0.7554, 3.8666, 1.6739, 1.0],
            ]
        )

    def from_XYZ(self, xyz: Tensor) -> Tensor:
        xyz_norm = xyz.reshape(-1, xyz.shape[-1])
        xyz_norm = xyz_norm / self.illuminant_XYZ
        ucs_crd = projective_transformation(xyz_norm, self.Q)
        ucs_crd = ucs_crd.view(xyz.shape[:-1] + ucs_crd.shape[-1:])
        return ucs_crd

    def to_XYZ(self, ucs_crd: Tensor) -> Tensor:
        inv_Q = torch.linalg.inv(self.Q)
        ucs_crd_2d = ucs_crd.reshape(-1, ucs_crd.shape[-1])
        xyz = projective_transformation(ucs_crd_2d, inv_Q)
        xyz = xyz * self.illuminant_XYZ
        xyz = xyz.reshape(ucs_crd.shape[:-1] + xyz.shape[-1:])
        return xyz


ref_prolab = ProLab()


def srgb2prolab(srgb: Tensor) -> Tensor:
    xyz: Tensor = color.rgb2xyz(srgb)
    prolab = ref_prolab.from_XYZ(xyz)
    return prolab


def srgb2lab(sRGB: Tensor) -> Tensor:
    XYZ: Tensor = color.rgb2xyz(sRGB)
    Lab = torch.from_numpy(colour.XYZ_to_Lab(XYZ))
    return Lab


def pixel_contrasts(
    image: Tensor, pixel: tuple[int, int], neighbors: tuple[Tensor, ...]
) -> Tensor:
    pixel_value = image[pixel]
    neighbor_values = image[neighbors]
    contrasts = torch.norm(pixel_value - neighbor_values, dim=1)
    return contrasts


def RMS_map(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
    n_pixel_neighbors: int = 1000,
    step: int = 10,
    sigma_rate: float = 0.25,
) -> Tensor:
    height, width, _ = img1.shape
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
            normalized_contrast_diffs = (
                (img1_contrasts - img2_contrasts)
                / 160
                # (img1_contrasts + img2_contrasts + 1) in weighted version
            )
            rms[i, j] = torch.sqrt(torch.mean(normalized_contrast_diffs**2))

    return rms


def RMS(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
    n_pixel_neighbors: int = 1000,
    step: int = 10,
    sigma_rate: float = 0.25,
):
    assert img1.dim() in (3, 4)
    assert img2.dim() in (3, 4)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    assert img1.shape[1] == 3
    assert img2.shape[1] == 3

    rms_maps = torch.empty((img1.shape[0]))
    for idx in range(img1.shape[0]):
        rms_maps[idx] = torch.mean(
            RMS_map(
                img1[idx].permute(1, 2, 0),
                img2[idx].permute(1, 2, 0),
                color_space,
                n_pixel_neighbors,
                step,
                sigma_rate,
            )
        )
    return rms_maps


def CD_map(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
    lightness_weight: int = 0,
) -> Tensor:
    from math import sqrt

    if color_space == "lab":
        lab1 = srgb2lab(img1)
        lab2 = srgb2lab(img2)

    elif color_space == "prolab":
        lab1 = srgb2prolab(img1)
        lab1[:, :, 0][lab1[:, :, 0] == 0] = 1
        lab1[:, :, 1] = lab1[:, :, 1] / lab1[:, :, 0]
        lab1[:, :, 2] = lab1[:, :, 2] / lab1[:, :, 0]

        lab2 = srgb2prolab(img2)
        lab2[:, :, 0][lab2[:, :, 0] == 0] = 1
        lab2[:, :, 1] = lab2[:, :, 1] / lab2[:, :, 0]
        lab2[:, :, 2] = lab2[:, :, 2] / lab2[:, :, 0]

    diff = lab1 - lab2
    weights = torch.tensor([sqrt(lightness_weight), 1, 1])
    weighted_diff = diff * weights
    chromatic_diff = torch.linalg.norm(weighted_diff, dim=2)
    return chromatic_diff


def CD(
    img1: Tensor,
    img2: Tensor,
    color_space: Literal["lab", "prolab"],
    lightness_weight: int = 0,
) -> Tensor:
    assert img1.dim() in (3, 4)
    assert img2.dim() in (3, 4)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    assert img1.shape[1] == 3
    assert img2.shape[1] == 3

    cd_maps = torch.empty((img1.shape[0]))
    for idx in range(img1.shape[0]):
        cd_maps[idx] = torch.mean(
            CD_map(
                img1[idx].permute(1, 2, 0),
                img2[idx].permute(1, 2, 0),
                color_space,
                lightness_weight,
            )
        )
    return cd_maps


def preprocess_img(img: Tensor) -> Tensor:
    img = img - img.min()
    return img / img.max()


if __name__ == "__main__":
    from skimage.io import imread

    img1_path = "/home/vkokhan/projects/dichrome/dichrome/trunk/rankendall_test/test_data/farup_simple/04.png"
    img1 = preprocess_img(
        torch.as_tensor(imread(img1_path))
    )  # .permute(2, 0, 1)
    img2_path = "/home/vkokhan/projects/dichrome/dichrome/trunk/rankendall_test/test_data/farup_simple_d/04.png"
    img2 = preprocess_img(
        torch.as_tensor(imread(img2_path))
    )  # .permute(2, 0, 1)

    color_space = "lab"

    print(CD(img1, img2, color_space=color_space))
