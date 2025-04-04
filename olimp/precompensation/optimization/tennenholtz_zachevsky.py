from typing import Literal, NamedTuple, Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion


class TennenholtzZachevskyParameters(NamedTuple):
    contrast_func_type: Literal["lin", "exp"] = "lin"
    sim_window_size: int = 11
    progress: Callable[[float], None] | None = None


def _create_sim_map(
    img_3ch_hsv: Tensor, img_2ch_hsv: Tensor, window: int
) -> Tensor:
    assert img_3ch_hsv.dim() == img_2ch_hsv.dim() == 3
    assert window % 2 == 1, f"window ({window}) must be odd"

    pad_h = window // 2
    pad_w = window // 2
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
                torch.norm(p_3ch - img_3ch_hsv[..., i : i + 1, j : j + 1])
            )
            mse_2ch = torch.sum(
                torch.norm(p_2ch - img_2ch_hsv[..., i : i + 1, j : j + 1])
            )
            sim_map[i, j] = mse_3ch - mse_2ch
    sim_map_stretched = (
        2 * (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min()) - 1
    )
    return sim_map_stretched[
        pad_h : sim_map_stretched.shape[0] - pad_h,
        pad_w : sim_map_stretched.shape[1] - pad_w,
    ]


def _contrast_func(
    sim_map: Tensor, params: Tensor, type: Literal["lin", "exp"]
) -> Tensor:
    if type == "lin":
        return params * sim_map
    elif type == "exp":
        return params[0] * sim_map * torch.exp(params[1] * torch.abs(sim_map))


def _set_range(img_3ch_hsv: Tensor, v_chan_stretched: Tensor) -> Tensor:
    img_3ch_hsv[2] = (v_chan_stretched - v_chan_stretched.min()) / (
        v_chan_stretched.max() - v_chan_stretched.min()
    ) * (img_3ch_hsv[2].max() - img_3ch_hsv[2].min()) + img_3ch_hsv[2].min()
    return img_3ch_hsv


def _hsv2rgb(hsv: Tensor) -> Tensor:
    from olimp.evaluation.cs.hsv import HSV

    return HSV().to_sRGB(hsv)


def _rgb2hsv(rgb: Tensor) -> Tensor:
    from olimp.evaluation.cs.hsv import HSV

    return HSV().from_sRGB(rgb)


def tennenholtz_zachevsky(
    img_3ch: Tensor,
    distortion: ColorBlindnessDistortion,
    parameters: TennenholtzZachevskyParameters = TennenholtzZachevskyParameters(),
) -> Tensor:
    """
    Tennenholtz-Zachevsky Natural Contrast Enhancement color blindness precompensation.

    .. image:: ../../_static/tennenholtz_zachevsky.svg
       :class: full-width
    """

    assert img_3ch.ndim == 3, img_3ch.ndim
    distortion_apply = distortion()

    img_2ch = distortion_apply(img_3ch[None])[0]

    img_3ch_hsv = _rgb2hsv(img_3ch.clone())
    img_2ch_hsv = _rgb2hsv(img_2ch.clone())

    sim_map_base = _create_sim_map(
        img_3ch_hsv, img_2ch_hsv, parameters.sim_window_size
    )

    if parameters.contrast_func_type == "lin":
        params_range = [torch.arange(0.01, 0.5, 0.05)]
    elif parameters.contrast_func_type == "exp":
        params_range = [torch.arange(0, 1, 0.1), torch.arange(0, 1, 0.1)]

    regularization_coef = 0.1
    optimal_error = 1000
    sim_map_curr = sim_map_base.clone()
    for params in params_range:
        for param in params:
            if parameters.progress is not None:
                parameters.progress(float(param / params[-1]))
            v_chan_stretched = img_3ch_hsv[2, ...] + _contrast_func(
                sim_map_base, param, type=parameters.contrast_func_type
            )
            img_3ch_hsv_v_stretched = _set_range(img_3ch_hsv, v_chan_stretched)

            img_2ch_enh = distortion_apply(
                _hsv2rgb(img_3ch_hsv_v_stretched.detach())[None]
            )[0]
            img_2ch_hsv_v_stretched = _rgb2hsv(img_2ch_enh.detach())

            sim_map_curr = _create_sim_map(
                img_3ch_hsv_v_stretched,
                img_2ch_hsv_v_stretched,
                window=parameters.sim_window_size,
            )
            curr_error = (sim_map_curr + 1).norm() + regularization_coef * (
                v_chan_stretched - img_3ch_hsv[2, ...]
            ).norm()
            if curr_error < optimal_error:
                optimal_error = curr_error
                optimal_params_img = img_3ch_hsv_v_stretched.clone()
    return _hsv2rgb(optimal_params_img)


def _demo():
    from .._demo_cvd import demo

    def demo_tennenholtz_zachevsky(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> tuple[torch.Tensor]:
        parameters = TennenholtzZachevskyParameters(progress=progress)
        return (tennenholtz_zachevsky(image[0], distortion, parameters)[None],)

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo(
        "Tennenholtz-Zachevsky",
        demo_tennenholtz_zachevsky,
        distortion=distortion,
    )


if __name__ == "__main__":
    _demo()
