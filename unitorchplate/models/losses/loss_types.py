from enum import auto

import torch
from strenum import StrEnum


class LossType(StrEnum):
    """Enum for the different loss types, this can be used to make the loss function configurable."""
    CUSTOM = auto()
    IN_ARCHITECTURE = auto()
    MSE = auto()

    @property
    def instance(self) -> torch.nn.Module | None:
        match self:
            case LossType.MSE:
                return torch.nn.MSELoss(reduction='none')
            case LossType.CUSTOM:
                return None
            case LossType.IN_ARCHITECTURE:
                return None
