from __future__ import annotations
from typing import Annotated, Literal
from .base import StrictModel
from pydantic import Field
from .dataset import PsfDataloaderConfig, ProgressCallback


class RefractionDistortionConfig(StrictModel):
    name: Literal["refraction_datasets"]
    psf: PsfDataloaderConfig

    def load(self, progress_callback: ProgressCallback):
        from .....simulate.refraction_distortion import RefractionDistortion

        dataset = self.psf.load(progress_callback)
        return dataset, RefractionDistortion()


class ColorBlindnessDistortionConfig(StrictModel):
    name: Literal["cvd"]
    blindness_type: Literal["deutan", "protan"]

    def load(self, progress_callback: ProgressCallback):
        from .....simulate.color_blindness_distortion import (
            ColorBlindnessDistortion,
        )

        return None, ColorBlindnessDistortion(self.blindness_type)


DistortionConfig = Annotated[
    RefractionDistortionConfig | ColorBlindnessDistortionConfig,
    Field(..., discriminator="name"),
]
