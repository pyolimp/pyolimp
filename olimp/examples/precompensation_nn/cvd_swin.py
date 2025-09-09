from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor

from olimp.precompensation._demo_cvd import demo as demo_cvd
from olimp.precompensation.nn.models.cvd_swin.cvd_swin_4channels import (
    CVDSwin4Channels,
)
from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion

if __name__ == "__main__":

    def demo_cvd_swin(
        image: Tensor,
        distortion: ColorBlindnessDistortion,
        progress: Callable[[float], None],
    ) -> tuple[torch.Tensor]:
        svd_swin = CVDSwin4Channels.from_path()
        image = svd_swin.preprocess(image, hue_angle_deg=torch.tensor([0.0]))
        progress(0.1)
        precompensation = svd_swin(image)
        progress(1.0)
        return (svd_swin.postprocess(precompensation[0]),)

    distortion = ColorBlindnessDistortion.from_type("protan")
    demo_cvd(
        "CVD-SWIN",
        demo_cvd_swin,
        distortion=distortion,
    )
