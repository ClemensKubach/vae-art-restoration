from __future__ import annotations
from enum import auto
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import ReduceLROnPlateau
from strenum import StrEnum
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from unitorchplate.models.modelmodule import ModelModule


class LrSchedulerType(StrEnum):
    """Enum for the different learning rate scheduler types, this can be used to make the learning rate scheduler
    configurable."""
    ReduceLROnPlateau = auto()
    NONE = auto()

    def instance(self, model: ModelModule) -> LRScheduler | ReduceLROnPlateau | None:
        match self:
            case LrSchedulerType.ReduceLROnPlateau:
                if model.config.lr_scheduler_config is None:
                    raise ValueError('lr_scheduler_config must be set for ReduceLROnPlateau')
                return ReduceLROnPlateau(
                    model.optimizer,
                    # monitor=model.config.monitor_metric,
                    mode=model.config.optimization_objective,
                    factor=model.config.lr_scheduler_config.factor,
                    patience=model.config.lr_scheduler_config.patience,
                    min_lr=model.config.lr_scheduler_config.min_lr
                )
            case LrSchedulerType.NONE:
                return None
