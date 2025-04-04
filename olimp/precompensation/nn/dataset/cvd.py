from __future__ import annotations
from . import BaseZenodoDataset, ImgPath, ProgressContext
from olimp.dataset.cvd import cvd as _cvd, Paths
from torch import Tensor


class CVDDataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_context: ProgressContext,
    ) -> dict[Paths, list[ImgPath]]:
        return _cvd(categories=categories, progress_context=progress_context)

    def __getitem__(self, index: int) -> Tensor:
        image = super().__getitem__(index)
        if image.shape[-3] == 4:  # fix cvd rgba images
            image = image[:3]
        return image
