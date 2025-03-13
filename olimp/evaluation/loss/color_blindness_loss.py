from __future__ import annotations
from typing import TypeAlias, Literal, Callable
import torch
from torch import Tensor

# from torchvision.transforms.v2 import Normalize
# from torchvision.transforms import Compose
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

    img_points1 = img[:, rand_width, rand_hight, :]
    img_points2 = img[:, rand_width1, rand_hight1, :]
    img1_diff = img_points1 - img_points2
    img1_diff = torch.sum(torch.abs(img1_diff), 2)

    img2_points1 = img2[:, rand_width, rand_hight, :]
    img2_points2 = img2[:, rand_width1, rand_hight1, :]

    img2_diff = img2_points1 - img2_points2
    img2_diff = torch.sum(torch.abs(img2_diff), 2)

    return img1_diff, img2_diff


class ColorBlindnessLoss:
    def __init__(
        self,
        lambda_ssim: float = 0.25,
        global_points: int = 3000,  # number of points to use to find global contrast
    ) -> None:
        self._global_points = global_points

        # self._trans_compose1: Callable[[Tensor], Tensor] = Compose(
        #     [Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))]
        # )
        # self._trans_compose2: Callable[[Tensor], Tensor] = Compose(
        #     [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # )
        # self._trans_compose3: Callable[[Tensor], Tensor] = Compose(
        #     [Normalize((0, 0, 0), (100, 128, 128))]
        # )

        self.rgb_norm_tf = transforms.Normalize(
            (-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)
        )
        self.lab_norm_tf = transforms.Normalize((0, 0, 0), (100, 128, 128))
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

        # return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))

    def __call__(
        self, image: Tensor, precompensated: Tensor, sim_f: ApplyDistortion
    ) -> Tensor:

        target_image = self.rgb_norm_tf(image)
        output_image = self.rgb_norm_tf(precompensated)

        # from minimg.view.view_client import connect; c = connect("B")
        # c.log("target_image", target_image.permute(0, 2, 3, 1)[0].detach().clone().cpu().numpy().copy())
        # c.log("output_image", output_image.permute(0, 2, 3, 1)[0].detach().clone().cpu().numpy().copy())


        # cvd_target = sim_f(target_image)
        cvd_output = sim_f(output_image)

        target_image_lab = torch.zeros_like(target_image)
        # for til, ti in zip(target_image_lab, target_image):
        #     til[:] = self._srgb2lab(ti)

        target_image_lab = self.lab_norm_tf(self._srgb2lab(target_image))
        output_image_lab = self.lab_norm_tf(self._srgb2lab(output_image))

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

        return (
            loss_contrast * (1 - self._lambda_ssim)
            + self._lambda_ssim * loss_ssim
        )
