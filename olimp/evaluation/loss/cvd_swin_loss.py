from __future__ import annotations
import torch
from torch import Tensor

import torchvision.transforms as transforms
from ..cs import D65 as D65_sRGB
from ..cs.cielab import CIELAB
from ..cs.srgb import sRGB
from .ssim import ContrastLoss, SSIMLoss

from olimp.simulate import ApplyDistortion


def _global_contrast_img_l1(
    img: Tensor, img2: Tensor, points_number: int = 5
) -> tuple[Tensor, Tensor]:
    img = img.permute(0, 2, 3, 1)
    img2 = img2.permute(0, 2, 3, 1)
    hight, width = img.shape[1], img.shape[2]

    rand_hight = torch.randint(0, hight, (points_number,))
    rand_width = torch.randint(0, width, (points_number,))

    rand_hight1 = torch.randint(0, hight, (points_number,))
    rand_width1 = torch.randint(0, width, (points_number,))

    img_points1 = img[:, rand_hight, rand_width, :]
    img_points2 = img[:, rand_hight1, rand_width1, :]
    img1_diff = img_points1 - img_points2
    img1_diff = torch.sum(torch.abs(img1_diff), 2)

    img2_points1 = img2[:, rand_hight, rand_width, :]
    img2_points2 = img2[:, rand_hight1, rand_width1, :]

    img2_diff = img2_points1 - img2_points2
    img2_diff = torch.sum(torch.abs(img2_diff), 2)

    return img1_diff, img2_diff


class CVDSwinLossBase:
    def __init__(
        self,
        lambda_ssim: float = 0.5,
        global_points: int = 3000,  # number of points to use to find global contrast
    ) -> None:
        self._global_points = global_points

        self.rgb_norm_tf = transforms.Normalize(
            (-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)
        )
        self.lab_norm_tf = transforms.Normalize((0, 0, 0), (1.00, 1.28, 1.28))
        self.model_input_norm_tf = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )

        self._contrast_loss = ContrastLoss()
        self._ssim_loss_funtion = SSIMLoss(kernel_size=11)
        self._lambda_ssim = lambda_ssim

    @staticmethod
    def _srgb2lab(srgb: Tensor) -> Tensor:
        output = []
        for i in srgb:
            output.append(CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(i)))
        return torch.stack(output, dim=0)

    def __call__(
        self, image: Tensor, precompensated: Tensor, sim_f: ApplyDistortion
    ) -> tuple[Tensor, Tensor, Tensor]:
        cvd_output = sim_f(precompensated)

        target_image_lab = self.lab_norm_tf(self._srgb2lab(image))
        output_image_lab = self.lab_norm_tf(self._srgb2lab(precompensated))

        cvd_output_lab = self.lab_norm_tf(self._srgb2lab(cvd_output))

        target_global_contrast, cvd_output_global_contrast = (
            _global_contrast_img_l1(
                target_image_lab, cvd_output_lab, self._global_points
            )
        )

        loss_contrast_local = self._contrast_loss(
            target_image_lab, cvd_output_lab
        )
        loss_contrast_global = torch.nn.L1Loss()(
            target_global_contrast, cvd_output_global_contrast
        )

        loss_contrast = loss_contrast_local + loss_contrast_global

        loss_ssim = self._ssim_loss_funtion(
            self.rgb_norm_tf(target_image_lab),
            self.rgb_norm_tf(output_image_lab),
        )
        loss = (
            loss_contrast * (1 - self._lambda_ssim)
            + self._lambda_ssim * loss_ssim
        )

        return loss, loss_contrast, loss_ssim


class CVDSwinLoss(CVDSwinLossBase):
    def __call__(
        self, image: Tensor, precompensated: Tensor, sim_f: ApplyDistortion
    ):
        return super().__call__(image, precompensated, sim_f)[0]
