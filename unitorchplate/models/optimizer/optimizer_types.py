from __future__ import annotations
from enum import auto
from typing import TYPE_CHECKING

from strenum import StrEnum
from torch import optim
from torch.optim import Optimizer

if TYPE_CHECKING:
    from unitorchplate.models.modelmodule import ModelModule


class OptimizerType(StrEnum):
    """Enum for the different optimizer types, this can be used to make the optimizer configurable."""
    ADAM = auto()
    ADAMW = auto()

    def instance(self, model: ModelModule) -> Optimizer:
        match self:
            case OptimizerType.ADAM:
                return optim.Adam(model.parameters(), lr=model.learning_rate)
            case  OptimizerType.ADAMW:
                return optim.AdamW(model.parameters(), lr=model.learning_rate)
