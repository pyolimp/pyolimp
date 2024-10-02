from __future__ import annotations
from typing import Callable, Generic, TypeVar

from torch import Tensor
from torch.utils.data import Dataset
from olimp.dataset._zenodo import ImgPath
from olimp.dataset import read_img_path

SubPath = TypeVar("SubPath", covariant=True)


class BaseZenodoDataset(Dataset[Tensor], Generic[SubPath]):
    def __init__(self, subsets: set[SubPath] | None):
        if subsets is None:
            subsets = getattr(self, "subsets", None)
            if not subsets:
                raise ValueError("Specify subsets or use predefined classes")

        dataset = self.create_dataset(
            categories=subsets, progress_callback=None
        )
        self._items = [item for subset in subsets for item in dataset[subset]]

    def create_dataset(
        self,
        categories: set[SubPath],
        progress_callback: Callable[[str, float], None] | None,
    ) -> dict[SubPath, list[ImgPath]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tensor:
        return read_img_path(self._items[index])

    def __len__(self):
        return len(self._items)
