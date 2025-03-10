from __future__ import annotations
from typing import NewType, Callable, TypeAlias
from pathlib import Path
from torch._prims_common import DeviceLikeType
import numpy as np
from torch import Tensor, tensor
from torchvision.io import read_image


ImgPath = NewType("ImgPath", Path)
ProgressCallback: TypeAlias = Callable[[str, float], None] | None


def read_img_path(path: ImgPath, device: DeviceLikeType = "cpu") -> Tensor:
    """
    Default device is "cpu" because it's the torch way
    """
    if path.suffix == ".csv":
        return tensor(
            np.loadtxt(path, delimiter=",", dtype=np.float32),
            device=device,
        ).unsqueeze(0)
    try:
        return read_image(str(path)).to(device=device)
    except Exception as e:
        raise ValueError(f"bad image file: {path}") from e
