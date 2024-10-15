from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion


def create_sim_map(
    img_3ch_hsv: Tensor, img_2ch_hsv: Tensor, window: tuple[int, int]
) -> Tensor:
    assert img_3ch_hsv.dim() == img_2ch_hsv.dim() == 3

    pad_h = window[0] // 2
    pad_w = window[1] // 2
    img_3ch_hsv = F.pad(
        img_3ch_hsv, pad=(pad_w, pad_w, pad_h, pad_h), mode="reflect"
    )
    img_2ch_hsv = F.pad(
        img_2ch_hsv, pad=(pad_w, pad_w, pad_h, pad_h), mode="reflect"
    )

    sim_map = torch.zeros(img_3ch_hsv.shape[1:])
    for i in range(pad_h, img_3ch_hsv.shape[1] - pad_h):
        for j in range(pad_w, img_3ch_hsv.shape[2] - pad_w):
            p_3ch = img_3ch_hsv[
                :, i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1
            ]
            p_2ch = img_2ch_hsv[
                :, i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1
            ]
            mse_3ch = torch.sum(
                torch.norm(
                    p_3ch
                    - img_3ch_hsv[..., i, j]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, window[0], window[1])
                )
            )
            mse_2ch = torch.sum(
                torch.norm(
                    p_2ch
                    - img_2ch_hsv[..., i, j]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, window[0], window[1])
                )
            )
            sim_map[i, j] = mse_3ch - mse_2ch
    sim_map_stretched = (
        2 * (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min()) - 1
    )
    return sim_map_stretched[
        pad_h : sim_map_stretched.shape[0] - pad_h,
        pad_w : sim_map_stretched.shape[1] - pad_w,
    ]


def contrast_func(
    sim_map: Tensor, params: Tensor, type: Literal["lin", "exp"]
) -> Tensor:
    if type == "lin":
        return params * sim_map
    elif type == "exp":
        return params[0] * sim_map * torch.exp(params[1] * torch.abs(sim_map))


def set_range(img_3ch_hsv: Tensor, v_chan_stretched: Tensor) -> Tensor:
    img_3ch_hsv[2] = (v_chan_stretched - v_chan_stretched.min()) / (
        v_chan_stretched.max() - v_chan_stretched.min()
    ) * (img_3ch_hsv[2].max() - img_3ch_hsv[2].min()) + img_3ch_hsv[2].min()
    return img_3ch_hsv


def tennenholtz_zachevsky(
    img_3ch: Tensor,
    distortion: ColorBlindnessDistortion,
    contrast_func_type: Literal["lin", "exp"],
    sim_window_size: int = 21,
) -> Tensor:
    """
    Tennenholtz-Zachevsky Natural Contrast Enhancement color blindness precompensation.
    """
    from skimage.color import rgb2hsv, hsv2rgb

    img_2ch = distortion(img_3ch)[0]

    img_3ch_hsv = torch.as_tensor(rgb2hsv(img_3ch.permute(1, 2, 0))).permute(
        2, 0, 1
    )
    img_2ch_hsv = torch.as_tensor(rgb2hsv(img_2ch.permute(1, 2, 0))).permute(
        2, 0, 1
    )

    sim_map_base = create_sim_map(
        img_3ch_hsv, img_2ch_hsv, window=(sim_window_size, sim_window_size)
    )

    if contrast_func_type == "lin":
        params_range = [torch.arange(0.01, 0.5, 0.05)]
    elif contrast_func_type == "exp":
        params_range = [torch.arange(0, 1, 0.1), torch.arange(0, 1, 0.1)]

    regularization_coef = 0.1
    optimal_error = 1000
    sim_map_curr = sim_map_base.clone()
    for params in params_range:
        for param in params:

            v_chan_streched = img_3ch_hsv[2, ...] + contrast_func(
                sim_map_base, param, type=contrast_func_type
            )
            img_3ch_hsv_v_streched = set_range(img_3ch_hsv, v_chan_streched)

            img_2ch_enh = distortion(
                torch.as_tensor(
                    hsv2rgb(img_3ch_hsv_v_streched.permute(1, 2, 0))
                ).permute(2, 0, 1)
            )[0]
            img_2ch_hsv_v_streched = torch.as_tensor(
                rgb2hsv(img_2ch_enh.permute(1, 2, 0))
            ).permute(2, 0, 1)

            sim_map_curr = create_sim_map(
                img_3ch_hsv_v_streched,
                img_2ch_hsv_v_streched,
                window=(sim_window_size, sim_window_size),
            )
            curr_error = (sim_map_curr + 1).norm() + regularization_coef * (
                v_chan_streched - img_3ch_hsv[2, ...]
            ).norm()
            if curr_error < optimal_error:
                optimal_error = curr_error
                optimal_params_img = img_3ch_hsv_v_streched.clone()
    return torch.as_tensor(
        hsv2rgb(optimal_params_img.permute(1, 2, 0))
    ).permute(2, 0, 1)
