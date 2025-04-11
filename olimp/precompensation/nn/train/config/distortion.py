from __future__ import annotations
from typing import Annotated, Literal
from .base import StrictModel
from pydantic import Field, model_validator
from .dataset import PsfDataloaderConfig, ProgressContext
from ballfish import DistributionParams


class RefractionDistortionConfig(StrictModel):
    name: Literal["refraction_datasets"]
    psf: PsfDataloaderConfig

    def load(self, progress_context: ProgressContext):
        from .....simulate.refraction_distortion import RefractionDistortion

        dataset = self.psf.load(progress_context)
        return dataset, RefractionDistortion()


class ColorBlindnessDistortionConfig(StrictModel):
    name: Literal["cvd"]
    blindness_type: Literal["deutan", "protan", "tritan"] | None = None
    hue_angle_deg: DistributionParams | None = None

    @model_validator(mode="after")
    def check_only_one_is_set(self):
        both_off = self.blindness_type is None and self.hue_angle_deg is None
        both_on = (
            self.blindness_type is not None and self.hue_angle_deg is not None
        )
        if both_off or both_on:
            raise ValueError(
                f"{'Only one' if both_on else 'One'} of `blindness_type` "
                f"{'and' if both_off else 'or'} `hue_angle_deg` must be set"
            )
        return self

    def load(self, progress_context: ProgressContext):
        from .....simulate.color_blindness_distortion import (
            ColorBlindnessDistortion,
        )

        if isinstance(self.hue_angle_deg, dict):  # must be distortion
            from ...dataset.value import ValueDataset

            dataset = ValueDataset(self.hue_angle_deg)
            return dataset, ColorBlindnessDistortion(None)
        if self.hue_angle_deg is not None:
            distortion = ColorBlindnessDistortion(self.hue_angle_deg)
        else:
            assert self.blindness_type is not None, "Programmer error"
            distortion = ColorBlindnessDistortion.from_type(
                self.blindness_type
            )
        return None, distortion


DistortionConfig = Annotated[
    RefractionDistortionConfig | ColorBlindnessDistortionConfig,
    Field(..., discriminator="name"),
]
