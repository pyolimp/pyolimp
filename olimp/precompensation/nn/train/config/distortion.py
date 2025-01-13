from __future__ import annotations
from typing import Annotated, Literal
from .base import StrictModel
from pydantic import Field
from .dataset import PsfDataloaderConfig


class RefractionDistortionConfig(StrictModel):
    name: Literal["refraction_datasets"]
    psf: PsfDataloaderConfig

    def load(self):
        from .....simulate.refraction_distortion import RefractionDistortion

        dataset = self.psf.load()
        return dataset, RefractionDistortion


DistortionConfig = Annotated[
    RefractionDistortionConfig,
    Field(..., discriminator="name"),
]
