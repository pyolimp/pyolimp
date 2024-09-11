from __future__ import annotations
from olimp.dataset.sca_2023 import sca_2023 as _sca_2023, Paths

from torch import Tensor
from torch.utils.data import Dataset


class SCA2023Dataset(Dataset[Tensor]):
    def __init__(self, subsets: set[Paths] | None):
        if subsets is None:
            subsets = getattr(self, "subsets", None)
            if not subsets:
                raise ValueError("Specify subsets or use predefined classes")

        dataset = _sca_2023(categories=subsets)
        self._items = [item for subset in subsets for item in dataset[subset]]

    def __getitem__(self, index: int):
        return self._items[index].data()

    def __len__(self):
        return len(self._items)


class SCA2023DatasetImages(SCA2023Dataset):
    subsets: set[Paths] = {"Images"}


class SCA2023DatasetPSF(SCA2023Dataset):
    subsets: set[Paths] = {"PSFs"}
