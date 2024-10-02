from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field
import torch
from .base import StrictModel


class AdamConfig(StrictModel):
    name: Literal["Adam"]
    learning_rate: float = 0.00001
    eps: float = 1e-8

    def load(self):
        def optimizer(model: torch.nn.Module):
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        return optimizer


class SGDConfig(StrictModel):
    name: Literal["SGD"]
    learning_rate: float = 0.00001

    def load(self):

        def optimizer(model: torch.nn.Module):
            return torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        return optimizer


Optimizer = Annotated[AdamConfig | SGDConfig, Field(..., discriminator="name")]
