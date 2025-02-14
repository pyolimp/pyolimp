from __future__ import annotations
from typing import TypeAlias, Literal, Callable
import torch
from torch import Tensor
from torchvision.transforms.v2 import Normalize
from torchvision.transforms import Compose
from ..cs import D65 as D65_sRGB
from ..cs.cielab import CIELAB
from ..cs.srgb import sRGB
from .ssim import ContrastLoss, SSIMLoss


CBType: TypeAlias = Literal["protan", "deutan"]
Degree: TypeAlias = Literal[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


CVD_MATRIX = {
    "protan": {
        10: (
            (0.856167, 0.182038, -0.038205),
            (0.029342, 0.955115, 0.015544),
            (-0.002880, -0.001563, 1.004443),
        ),
        20: (
            (0.734766, 0.334872, -0.069637),
            (0.051840, 0.919198, 0.028963),
            (-0.004928, -0.004209, 1.009137),
        ),
        30: (
            (0.630323, 0.465641, -0.095964),
            (0.069181, 0.890046, 0.040773),
            (-0.006308, -0.007724, 1.014032),
        ),
        40: (
            (0.539009, 0.579343, -0.118352),
            (0.082546, 0.866121, 0.051332),
            (-0.007136, -0.011959, 1.019095),
        ),
        50: (
            (0.458064, 0.679578, -0.137642),
            (0.092785, 0.846313, 0.060902),
            (-0.007494, -0.016807, 1.024301),
        ),
        60: (
            (0.385450, 0.769005, -0.154455),
            (0.100526, 0.829802, 0.069673),
            (-0.007442, -0.022190, 1.029632),
        ),
        70: (
            (0.319627, 0.849633, -0.169261),
            (0.106241, 0.815969, 0.077790),
            (-0.007025, -0.028051, 1.035076),
        ),
        80: (
            (0.259411, 0.923008, -0.182420),
            (0.110296, 0.804340, 0.085364),
            (-0.006276, -0.034346, 1.040622),
        ),
        90: (
            (0.203876, 0.990338, -0.194214),
            (0.112975, 0.794542, 0.092483),
            (-0.005222, -0.041043, 1.046265),
        ),
        100: (
            (0.152286, 1.052583, -0.204868),
            (0.114503, 0.786281, 0.099216),
            (-0.003882, -0.048116, 1.05199),
        ),
    },
    "deutan": {
        10: (
            (0.866435, 0.177704, -0.044139),
            (0.049567, 0.939063, 0.011370),
            (-0.003453, 0.007233, 0.996220),
        ),
        20: (
            (0.760729, 0.319078, -0.079807),
            (0.090568, 0.889315, 0.020117),
            (-0.006027, 0.013325, 0.992702),
        ),
        30: (
            (0.675425, 0.433850, -0.109275),
            (0.125303, 0.847755, 0.026942),
            (-0.007950, 0.018572, 0.989378),
        ),
        40: (
            (0.605511, 0.528560, -0.134071),
            (0.155318, 0.812366, 0.032316),
            (-0.009376, 0.023176, 0.986200),
        ),
        50: (
            (0.547494, 0.607765, -0.155259),
            (0.181692, 0.781742, 0.036566),
            (-0.010410, 0.027275, 0.983136),
        ),
        60: (
            (0.498864, 0.674741, -0.173604),
            (0.205199, 0.754872, 0.039929),
            (-0.011131, 0.030969, 0.980162),
        ),
        70: (
            (0.457771, 0.731899, -0.189670),
            (0.226409, 0.731012, 0.042579),
            (-0.011595, 0.034333, 0.977261),
        ),
        80: (
            (0.422823, 0.781057, -0.203881),
            (0.245752, 0.709602, 0.044646),
            (-0.011843, 0.037423, 0.974421),
        ),
        90: (
            (0.392952, 0.823610, -0.216562),
            (0.263559, 0.690210, 0.046232),
            (-0.011910, 0.040281, 0.971630),
        ),
        100: (
            (0.367322, 0.860646, -0.227968),
            (0.280085, 0.672501, 0.047413),
            (-0.011820, 0.042940, 0.968881),
        ),
    },
}


def _srgb2lab(srgb: Tensor) -> Tensor:
    srgb = (srgb / 255.0).clip(min=0, max=1.0)
    return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def _lab2srgb(lab: Tensor) -> Tensor:
    return (
        sRGB().from_XYZ(CIELAB(D65_sRGB).to_XYZ(lab).clip(min=0.0, max=1.0))
    ) * 255


def _global_contrast_img_l1(
    img: Tensor, img2: Tensor, points_number: int = 5
) -> tuple[Tensor, Tensor]:
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

        self._trans_compose1: Callable[[Tensor], Tensor] = Compose(
            [Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))]
        )
        self._trans_compose2: Callable[[Tensor], Tensor] = Compose(
            [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self._trans_compose3: Callable[[Tensor], Tensor] = Compose(
            [Normalize((0, 0, 0), (100, 128, 128))]
        )
        self._contrast_loss = ContrastLoss()
        self._ssim_loss_funtion = SSIMLoss(kernel_size=11)
        self._lambda_ssim = lambda_ssim

    def set_cb_type(self, cb_type: CBType, degree: Degree = 100) -> None:
        self._cb_type = cb_type
        t_tensor = (
            torch.tensor(CVD_MATRIX[cb_type][degree]).type(torch.float32).T
        )
        self.t_tensor = t_tensor.unsqueeze(0)

    def _cvd_simulation_tensors(
        self,
        img: Tensor,
    ) -> Tensor:
        """
        from https://github.com/Ligeng-c/CVD_swin/blob/07271c5bc0f89068e8fc8cfa57cec0a9b8800549/cvd_function.py#L166-L193
        """
        t_tensor = self.t_tensor.repeat([img.shape[0], 1, 1]).to(
            device=img.device
        )

        h, w = img.shape[2], img.shape[3]
        img = img.view([-1, 3, h * w])

        img = img.permute(0, 2, 1)  # B H*W C

        cvd_img = torch.bmm(img, t_tensor)
        cvd_img = cvd_img.permute(0, 2, 1)  # B  C  H*W
        cvd_img = cvd_img.view([-1, 3, h, w])

        out_put = cvd_img.clip(min=0.0, max=1.0)
        return out_put

    def __call__(self, image: Tensor, precompensated: Tensor) -> Tensor:
        image = self._trans_compose1(image)
        precompensated = self._trans_compose1(precompensated)

        cvd_precompensated = self._cvd_simulation_tensors(precompensated)

        cvd_precompensated = _srgb2lab(cvd_precompensated)
        image = _srgb2lab(image)

        precompensated = self._trans_compose3(_lab2srgb(precompensated))
        cvd_precompensated = self._trans_compose3(cvd_precompensated)
        image = self._trans_compose3(image)

        loss_contrast = self._contrast_loss(image, cvd_precompensated)

        g_contrast_1, g_contrast_2 = _global_contrast_img_l1(
            image, cvd_precompensated, self._global_points
        )

        loss_contrast_global = torch.nn.L1Loss()(g_contrast_1, g_contrast_2)

        loss_ssim = self._ssim_loss_funtion(
            self._trans_compose2(image), self._trans_compose2(precompensated)
        )

        lambda_ssim = self._lambda_ssim

        color_blindness_loss = (loss_contrast + 1 * loss_contrast_global) * (
            1 - lambda_ssim
        ) + lambda_ssim * loss_ssim
        return color_blindness_loss
