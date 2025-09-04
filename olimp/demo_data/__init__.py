from typing import TypedDict
from pathlib import Path
from torchvision.io import read_image
from torch import Tensor


def horse() -> Tensor:
    return read_image(str(Path(__file__).with_name("horse.jpg")))


def ishihara() -> Tensor:
    return read_image(str(Path(__file__).with_name("73.png")))


class PSF(TypedDict):
    psf: Tensor
    S: float
    C: float
    A: float


def _psf(S: float, C: float, A: float) -> PSF:
    from olimp.simulate.psf_sca import PSFSCA

    psf = PSFSCA(512, 512)(
        sphere_dpt=S,
        cylinder_dpt=C,
        angle_rad=A,
        pupil_diameter_mm=4.0,
        am2px=0.001,
    )
    ret: PSF = {
        "psf": psf,
        "S": S,
        "C": C,
        "A": A,
    }
    return ret


def psf() -> PSF:
    S, C, A = -2.269, -1.019, 2.8099800957108707
    return _psf(S=S, C=C, A=A)


def psf2() -> PSF:
    S, C, A = -1.167, 0.083, -0.4363323129985824
    return _psf(S=S, C=C, A=A)
