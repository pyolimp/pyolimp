from __future__ import annotations
from . import BaseZenodoDataset, ImgPath, ProgressContext
from olimp.dataset.olimp import olimp as _olimp, Paths


class OlimpDataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_context: ProgressContext,
    ) -> dict[Paths, list[ImgPath]]:
        return _olimp(categories=categories, progress_context=progress_context)
