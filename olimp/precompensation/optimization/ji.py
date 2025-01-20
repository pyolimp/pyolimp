from __future__ import annotations
from typing import NamedTuple, TypedDict, Callable
import torch
import torch.nn.functional as F
from olimp.processing import conv
from olimp.precompensation.basic.huang import huang
from olimp.evaluation.loss.piq import MultiScaleSSIMLoss
from torch import Tensor
from typing import Tuple


class DebugInfo(TypedDict):
    loss_step: list[float]
    precomp: torch.Tensor


class JiParametrs(NamedTuple):
    lr_m: float = 1.0
    lr_tau: float = 1e-2

    m0: float = 1.0
    gap: float = 1e-1
    gap_iter: float = 1e-1  # stop criterian

    ibf_param: float = 1e-3
    alpha: float = 1e-1
    num_of_iter: int = 100

    partition_step: float = 1e-3


def inverse_blur_filtering(
    image: Tensor, psf: Tensor, k: float = 0.01
) -> Tensor:
    is_rgb = image.shape[1] == 3

    if not is_rgb:
        return huang(image, psf, k)

    ibf = torch.zeros(image.shape)
    for i in range(image.shape[1]):
        ibf[:, i, ...] = huang(image[:, i, ...], psf, k)
    return ibf.requires_grad_(True)


def linear_normalize(image: Tensor) -> Tensor:
    return (image - torch.min(image)) / (torch.max(image) - torch.min(image))


def histogram_maximum(image: Tensor, bins_number: int = 10000) -> Tensor:
    image = linear_normalize(image)

    hist = torch.histc(image, bins=bins_number, min=0, max=1)
    mode = torch.argmax(hist) / bins_number

    return mode


def equivalent_ringing_free(image: Tensor, m: Tensor, mode: Tensor) -> Tensor:
    return torch.clip(m * (image - mode) + mode, min=0, max=1)


def bezier_curve(
    t: Tensor,
    mode: Tensor,
    m: Tensor,
    tau_plus: Tensor,
    tau_minus: Tensor,
    P_minus: Tensor = torch.Tensor([0, 0]),
    P_plus: Tensor = torch.Tensor([1, 1]),
) -> Tuple[Tensor, Tensor]:
    # Define theta
    theta = torch.arctan(1 / m)

    # Limits on tau_plus, tau_minus
    # tau_plus = torch.clamp(tau_plus, 0, ((1 - mode) / torch.cos(theta.clone())).item())
    # tau_minus = torch.clamp(tau_minus, 0, (mode / torch.cos(theta.clone())).item())

    # Define Q and P
    Q_minus = torch.Tensor(
        [
            mode - tau_minus * torch.sin(theta),
            mode - tau_minus * torch.cos(theta),
        ]
    )
    Q_plus = torch.Tensor(
        [
            mode + tau_plus * torch.sin(theta),
            mode + tau_plus * torch.cos(theta),
        ]
    )

    P = torch.Tensor([mode, mode])

    # Calculate curve
    Bx_minus = (
        torch.pow(1 - t, 2) * P_minus[0]
        + 2 * t * (1 - t) * Q_minus[0]
        + torch.pow(t, 2) * P[0]
    )
    By_minus = (
        torch.pow(1 - t, 2) * P_minus[1]
        + 2 * t * (1 - t) * Q_minus[1]
        + torch.pow(t, 2) * P[1]
    )

    Bx_plus = (
        torch.pow(1 - t, 2) * P[0]
        + 2 * t * (1 - t) * Q_plus[0]
        + torch.pow(t, 2) * P_plus[0]
    )
    By_plus = (
        torch.pow(1 - t, 2) * P[1]
        + 2 * t * (1 - t) * Q_plus[1]
        + torch.pow(t, 2) * P_plus[1]
    )

    Bx = torch.concatenate([Bx_minus, Bx_plus])
    By = torch.concatenate([By_minus, By_plus])

    return Bx, By


def mapping_function(
    t: Tensor,
    image: Tensor,
    m: Tensor,
    tau_plus: Tensor,
    tau_minus: Tensor,
    mode: Tensor,
) -> Tensor:
    # Bezier curve
    Bx, By = bezier_curve(t, mode, m, tau_plus, tau_minus)
    length = Bx.size(dim=0)

    # Use torch.searchsorted to find which interval each pixel belongs to
    indices = torch.searchsorted(Bx, image, right=False)

    # Clip indices to ensure they are within the valid range
    indices = torch.clamp(indices, 0, length - 1)

    # Use the indices to select the corresponding By values
    res = By[indices]

    return res


def ji(
    image: Tensor,
    psf: Tensor,
    parameters: JiParametrs = JiParametrs(),
) -> Tensor:
    t = torch.arange(0, 1, parameters.partition_step)

    loss_func = MultiScaleSSIMLoss()
    prev_total_loss = torch.tensor(float("inf"))

    loss_step: list[float] = []

    # Calculating inverse blur and it normalisation
    inverse_blur = inverse_blur_filtering(image, psf, parameters.ibf_param)
    ibf_linear_normalise = linear_normalize(inverse_blur)

    # Calculate mode
    mode = histogram_maximum(ibf_linear_normalise)

    m0 = torch.tensor([parameters.m0]).requires_grad_(True)
    tau_plus0 = torch.tensor([0.25]).requires_grad_(True)
    tau_minus0 = torch.tensor([0.25]).requires_grad_(True)

    for k in range(parameters.num_of_iter):  # type: ignore
        # Solving tau-subproblem with fixed m
        prev_loss = torch.tensor(float("inf"))

        tau_plus = tau_plus0.clone().detach().requires_grad_(True)
        tau_minus = tau_minus0.clone().detach().requires_grad_(True)
        optimizer_tau = torch.optim.Adam(
            [tau_plus, tau_minus], lr=parameters.lr_tau
        )

        for i in range(1000):  # type: ignore
            optimizer_tau.zero_grad()

            # Tone mapped image
            mapped = mapping_function(
                t, ibf_linear_normalise, m0, tau_plus, tau_minus, mode
            )

            mapped_conv = conv(mapped, psf)

            # MS-SSIM + original image
            loss = loss_func(
                mapped_conv,
                image,
            )

            loss.backward()
            optimizer_tau.step()  # type: ignore

            if torch.abs(prev_loss - loss).item() < parameters.gap:
                break

            prev_loss = loss

            # Update tau
            # tau_plus0 = tau_plus.detach()
            # tau_minus0 = tau_minus.detach()

            theta = torch.arctan(1 / m0)

            tau_plus0 = torch.clamp(
                tau_plus.clone().detach(),
                0,
                ((1 - mode) / torch.cos(theta.clone())).item(),
            )
            tau_minus0 = torch.clamp(
                tau_minus.clone().detach(),
                0,
                (mode / torch.cos(theta.clone())).item(),
            )

            # Solving m-subproblem with fixed tau
            prev_loss = torch.tensor(float("inf"))

            m = torch.tensor(
                m0.clone().detach(), dtype=torch.float32, requires_grad=True
            )
            optimizer_m = torch.optim.Adam([m], lr=parameters.lr_m)  # type: ignore

            for i in range(1000):  # type: ignore
                optimizer_m.zero_grad()

                # Tone mapped image
                mapped = mapping_function(
                    t, ibf_linear_normalise, m, tau_plus0, tau_minus0, mode
                )
                mapped_conv = conv(mapped, psf)

                loss = loss_func(mapped_conv, image)

                loss.backward()
                optimizer_m.step()  # type: ignore

                if torch.abs(prev_loss - loss).item() < parameters.gap:
                    break

                prev_loss = loss

            # Update m
            m0 = m.detach()
            loss_step.append(loss.item())

            print(loss.item())

            # Stop criteria
            if abs(prev_total_loss - loss.item()) < parameters.gap_iter:
                break

            prev_total_loss = loss.item()

        m = m0
        mode = mode.detach()
        tau_plus = tau_plus0.detach()
        tau_minus = tau_minus0.detach()

        return mapped


if __name__ == "__main__":
    import numpy as np

    import torchvision
    from olimp.processing import conv
    from torchvision.transforms.v2 import Resize, Grayscale

    import matplotlib.pyplot as plt

    psf_info = np.load("/home/alkzir/projects/pyolimp/tests/test_data/psf.npz")
    img = torchvision.io.read_image(
        "/home/alkzir/projects/pyolimp/tests/test_data/horse.jpg"
    )
    img = img / 255.0
    img = Resize((512, 512))(img)
    img = img.unsqueeze(0).requires_grad_(True)

    psf = (
        torch.fft.fftshift(torch.tensor(psf_info["psf"]).to(torch.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    ).requires_grad_(True)

    print(psf.shape)
    print(img.shape)

    plt.imshow(ji(img, psf, JiParametrs())[0][0])
    plt.show()
