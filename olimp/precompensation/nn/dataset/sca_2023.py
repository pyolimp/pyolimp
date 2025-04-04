from __future__ import annotations
from . import BaseZenodoDataset, ImgPath, ProgressContext
from olimp.dataset.sca_2023 import sca_2023 as _sca_2023, Paths


class SCA2023Dataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_context: ProgressContext,
    ) -> dict[Paths, list[ImgPath]]:
        return _sca_2023(
            categories=categories, progress_context=progress_context
        )
