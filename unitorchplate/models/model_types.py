from __future__ import annotations
from enum import auto
from typing import TYPE_CHECKING

import torch.nn
from strenum import StrEnum

if TYPE_CHECKING:
    from unitorchplate.models.modelmodule import ModelConfig, ModelModule


class ModelTypes(StrEnum):
    """Enum for the different model types, this can be used to make the model configurable."""
    CUSTOM = auto()

    def cls(self) -> type[ModelModule]:
        """Return the class of the selected model type."""
        match self:
            case ModelTypes.CUSTOM:
                raise NotImplementedError("CUSTOM model type is not implemented")

    def instance(self, config: ModelConfig, preprocessing: torch.nn.Module) -> ModelModule | None:
        match self:
            case ModelTypes.CUSTOM:
                return None
