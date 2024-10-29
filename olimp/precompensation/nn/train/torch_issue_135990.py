from __future__ import annotations
from typing import Iterator
import torch
from torch.utils.data import RandomSampler


class RandomSamplerCPU(RandomSampler):
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        assert not self.replacement
        for _ in range(self.num_samples // n):
            yield from torch.randperm(
                n, generator=generator, device="cpu"
            ).tolist()
        yield from torch.randperm(
            n, generator=generator, device="cpu"
        ).tolist()[: self.num_samples % n]
