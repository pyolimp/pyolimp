from __future__ import annotations
from typing import Literal

PyOlimpHF = (
    Literal[
        "hf://RVI/cvae.pth",
        "hf://RVI/dwdn.pt",
        "hf://RVI/unet-efficientnet-b0.pth",
        "hf://RVI/unetvae.pth",
        "hf://RVI/usrnet.pth",
        "hf://RVI/vae.pth",
        "hf://RVI/vdsr.pth",
        "hf://CVD/cvd_swin.pth",
        "hf://CVD/Generator_cnn_pathch4_844_48_3_nouplayer_server5.pth",
        "hf://CVD/Generator_transformer_pathch4_1_1.pth",
        "hf://CVD/Generator_transformer_pathch4_8_3_48_3.pth",
        "hf://CVD/Generator_transformer_pathch4_844_48_3.pth",
        "hf://CVD/cvd_swin_1channel.pth",
        "hf://tone_mapping/hdrnet_v0.pt",
    ]
    | str
)  # for suggestions only


def download_path(path: PyOlimpHF) -> str:
    """
    "hf://" protocol is a local pyolimp convention
    """
    if path.startswith("hf://"):
        filename = path[len("hf://") :]

        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id="pyolimp/pyolimp", filename=filename)
    else:
        from os.path import expanduser

        return expanduser(path)
