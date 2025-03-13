from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module
from itertools import repeat

from ..cs.cielab import CIELAB
# from ..cs.prolab import ProLab


def srgb2lab(srgb: Tensor) -> Tensor:   # load from another module (rms or chromaticity_difference???)
    return CIELAB(D65_sRGB).from_XYZ(sRGB().to_XYZ(srgb))


def get_perms_indices(Len):
    t1 = torch.zeros(g_mul * Len, dtype=torch.int32)
    t2 = torch.zeros(g_mul * Len, dtype=torch.int32)

    for ii in range(0, g_mul * Len, Len):
        t1[ii : ii + Len] = torch.randperm(Len)
        t2[ii : ii + Len] = torch.randperm(Len)
        # t1[ii : ii + Len] = np.random.permutation(Len)
        # t2[ii : ii + Len] = np.random.permutation(Len)
    return t1, t2

def color_init_w(img, t1, t2, cs):
    # reshaping check at the end of function
    breakpoint()
    # np.random.seed(0)         # seemed important. How implement?

    # imgLab = metrics_ex.cs_convert_img(img, color_space=cs)
    if cs == "lab":
        imgLab = srgb2lab(img)

    elif cs == "prolab":
        raise ValueError("implemente me")
    #     imgLab = srgb2prolab(img)
    
    imgV = imgLab.reshape(-1, 3)
    # colorContr = np.sqrt(((imgV[t1, :] - imgV[t2, :]) ** 2).sum(axis=1))
    colorContr = torch.sqrt(((imgV[t1, :] - imgV[t2, :]) ** 2).sum(axis=1))
    
    return colorContr

def get_init_w(img, t1, t2, cs):
    # reshaping as in color_init_w
    breakpoint()
    # CCPR
    # imgLab = metrics_ex.cs_convert_img(img, color_space=cs)
    if cs == "lab":
        imgLab = srgb2lab(img)
        
    imgV = imgLab.reshape(-1, 3)
    # colorContr = np.sqrt(((imgV[t1, :] - imgV[t2, :]) ** 2).sum(axis=1))
    colorContr = torch.sqrt(((imgV[t1, :] - imgV[t2, :]) ** 2).sum(axis=1))

    return colorContr

def CCPR_fun_fast(img0, img1, thr):
    img0_mask = img0 > thr
    img1_mask = img1[img0_mask] > thr
    num = img1_mask.sum()
    den = img0_mask.sum()
    den = 1 if den == 0 else den

    ccpr = num / den

    return ccpr

def CCFR_fun_fast(img0, img1, thr):
    img1_mask = img1 > thr
    img0_mask = img0[img1_mask] < thr

    num = img0_mask.sum()
    den = img1_mask.sum()
    den = 1 if den == 0 else den

    ccfr = 1 - num / den

    return ccfr

def worker(i, local_h, simulation, image):
    win_height_count = int((height - int(hw / 2) + 1) / g_stride) + 1
    win_width_count = int((width - int(hw / 2) + 1) / g_stride) + 1

    # arr0 = np.frombuffer(ccpr_o.get_obj(), dtype="float32")
    # arr1 = np.frombuffer(ccpr_s.get_obj(), dtype="float32")
    # b_ori = arr0.reshape(win_height_count, win_width_count, -1)
    # b_sim = arr1.reshape(win_height_count, win_width_count, -1)
    arr0 = ccpr_o    # check independence copy
    arr1 = ccpr_s    # check independence copy
    print("arr0: ", arr0[:5])
    print("ccpr_o: ", ccpr_o[:5])
    breakpoint()
    b_ori = torch.reshape(arr0, (win_height_count, win_width_count, -1))
    b_sim = torch.reshape(arr1, (win_height_count, win_width_count, -1))

    shift = int(hw / 2)

    for j, local_w in enumerate(range(shift, width, g_stride)):
        img = image[
            local_h - shift : local_h + shift + 1, local_w - shift : local_w + shift + 1
        ]
        sim = simulation[
            local_h - shift : local_h + shift + 1, local_w - shift : local_w + shift + 1
        ]
        # Len = sim.shape[0] * sim.shape[1]
        Len = sim.shape[1] * sim.shape[2]
        t1, t2 = get_perms_indices(Len)

        colorContr_o = color_init_w(img, t1, t2, cs=g_cs)
        colorContr_s = get_init_w(sim, t1, t2, cs=g_cs)

        b_ori[i, j, : colorContr_o.shape[0]] = colorContr_o
        b_sim[i, j, : colorContr_s.shape[0]] = colorContr_s
    return 1


def wEscore_map(
    simulation: Tensor, 
    image: Tensor, 
    color_space: Literal["lab", "prolab"],
    lp: int = 61,
    lf: int = 7,
    thr: int = 6,
) -> Tensor:

    assert lp % 2 == 1 or lf % 2 == 1, "lp and lf should be odd"

    _channels, h, w = simulation.shape
    # h, w = simulation.shape[:2]    # old simulation: [H, W, C] or [H, W, C, B]

    # CCPR
    # number of pairs to compare in each window
    mul = 5
    Len = lp * lp
    Len_thresh = mul * Len
    stride_p = int(lp / 2)
    win_height_count = int((h - int(lp / 2) + 1) / stride_p) + 1
    win_width_count = int((w - int(lp / 2) + 1) / stride_p) + 1

    CCPR_contr_o_lst = torch.zeros(Len_thresh * win_height_count * win_width_count, dtype=torch.float32)
    CCPR_contr_s_lst = torch.zeros(Len_thresh * win_height_count * win_width_count, dtype=torch.float32)

    # REMOVE GLOBAL PARAMS (remake to avoid global)
    global ccpr_o, ccpr_s, height, width, hw, g_stride, g_mul, g_cs
    ccpr_o = CCPR_contr_o_lst
    ccpr_s = CCPR_contr_s_lst
    height = h
    width = w
    hw = lp
    g_stride = stride_p
    g_mul = mul
    g_cs = color_space
    # END GLOBAL PARAMS

    # p.starmap(
    #     worker,
    #     zip(
    #         np.arange(win_height_count),
    #         np.arange(int(lp / 2), h, stride_p),
    #         repeat(simulation),
    #         repeat(image),
    #     ),
    # )
    # vvvvv first problem here vvvv
    map(
        worker,
        zip(
            torch.arange(win_height_count),
            torch.arange(int(lp / 2), h, stride_p),
            repeat(simulation),
            repeat(image),
        ),
    )
    breakpoint()

    sup = CCPR_contr_o_lst      # check independence copy
    CCPR_contr_o_lst = torch.reshape(sup, (win_height_count, win_width_count, Len_thresh))

    sup = CCPR_contr_s_lst      # check independence copy
    CCPR_contr_s_lst = torch.reshape(sup, (win_height_count, win_width_count, Len_thresh))

    ccpr = CCPR_fun_fast(CCPR_contr_o_lst, CCPR_contr_s_lst, thr)

    del lp, CCPR_contr_o_lst, CCPR_contr_s_lst

    # CCFR
    Len = lf * lf
    Len_thresh = mul * Len
    stride_f = int(lf / 2)
    win_height_count = int((h - int(lf / 2) + 1) / stride_f) + 1
    win_width_count = int((w - int(lf / 2) + 1) / stride_f) + 1

    CCFR_contr_o_lst = torch.zeros(Len_thresh * win_height_count * win_width_count, dtype=torch.float32)
    CCFR_contr_s_lst = torch.zeros(Len_thresh * win_height_count * win_width_count, dtype=torch.float32)

    # REMOVE GLOBAL PARAMS (remake to avoid global)
    global ccpr_o, ccpr_s, height, width, hw, g_stride, g_mul, g_cs
    ccpr_o = CCFR_contr_o_lst
    ccpr_s = CCFR_contr_s_lst
    height = h
    width = w
    hw = lf
    g_stride = stride_f
    g_mul = mul
    g_cs = color_space
    # END GLOBAL PARAMS
    # p = Pool(
    #     processes=processes,
    #     initializer=init,
    #     initargs=(
    #         CCFR_contr_o_lst,
    #         CCFR_contr_s_lst,
    #         h,
    #         w,
    #         lf,
    #         stride_f,
    #         mul,
    #         color_space,
    #     ),
    # )
    
    # p.starmap(
    #     worker,
    #     zip(
    #         np.arange(win_height_count),
    #         np.arange(int(lf / 2), h, stride_f),
    #         repeat(simulation),
    #         repeat(image),
    #     ),
    # )

    map(
        worker,
        zip(
            torch.arange(win_height_count),
            torch.arange(int(lf / 2), h, stride_f),
            repeat(simulation),
            repeat(image),
        ),
    )

    sup = CCFR_contr_o_lst   # check independence copy
    CCFR_contr_o_lst = torch.reshape(sup, (win_height_count, win_width_count, Len_thresh))

    sup = CCFR_contr_s_lst   # check independence copy
    CCFR_contr_s_lst = torch.reshape(sup, (win_height_count, win_width_count, Len_thresh))

    ccfr = CCFR_fun_fast(CCFR_contr_o_lst, CCFR_contr_s_lst, thr)

    eps = 1e-16
    wE_score = 2 * ccpr * ccfr / (ccpr + ccfr + eps)

    del CCFR_contr_o_lst, CCFR_contr_s_lst

    return wE_score


class wEscore(Module):
    _color_space: Literal["lab", "prolab"]   # only lab?

    def __init__(
        self,
        color_space: Literal["lab", "prolab"],
        lp: int = 61,
        lf: int = 7,
        thr: int = 6,   # check int/float
    ):
        super().__init__()
        self._color_space = color_space
        self._lp = lp
        self._lf = lf
        self._thr = thr

    def forward(self, img1: Tensor, img2: Tensor):
        assert img1.ndim == 4, img1.shape
        assert img2.ndim == 4, img2.shape

        assert img1.shape[1] == 3
        assert img2.shape[1] == 3

        wescore_maps = torch.empty((img1.shape[0]))
        for idx in range(img1.shape[0]):
            wescore_maps[idx] = wEscore_map(
                img1[idx],
                img2[idx],
                self._color_space,
                self._lp,
                self._lf,
                self._thr,
            )
        return wescore_maps
    

def demo(distortion: ColorBlindnessDistortion):
    from __future__ import annotations
    from typing import Literal, Callable
    import matplotlib.pyplot as plt
    import torch
    from torch import Tensor
    import torchvision
    from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion
    from torchvision.transforms.v2 import Resize
    from pathlib import Path

    from olimp.evaluation.loss.wescore import wEscore


    def test_wescore(distortion: ColorBlindnessDistortion = distortion):
        root = Path(__file__).parents[2]
        img = torchvision.io.read_image(root / "tests/test_data/73.png")[None]
        img = img / 255.0
        img = Resize((256, 256))(img)

        cvd_img = distortion()(img)
        loss_wEscore = wEscore(
                    "lab"
                )
        # loss_G = torch.mean(loss_rms(real_A, cvd_fake))
        res = loss_wEscore(cvd_img, img)
        print("wEscore: ", res)

    test_wescore()


if __name__ == "__main__":
    demo(distortion = ColorBlindnessDistortion("deutan"))