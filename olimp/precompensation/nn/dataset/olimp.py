from __future__ import annotations
from . import BaseZenodoDataset, ImgPath, ProgressCallback
from olimp.dataset.olimp import olimp as _olimp, Paths


class OlimpDataset(BaseZenodoDataset[Paths]):
    def create_dataset(
        self,
        categories: set[Paths],
        progress_callback: ProgressCallback,
    ) -> dict[Paths, list[ImgPath]]:
        return _olimp(
            categories=categories, progress_callback=progress_callback
        )
