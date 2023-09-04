from __future__ import annotations
from typing import TYPE_CHECKING

from strenum import StrEnum
from torch import Tensor
from torch.nn import Module

from unitorchplate.data.datamodules import DataModule

if TYPE_CHECKING:
    from unitorchplate.models.modelmodule import ModelConfig


class ConcatenatedTransformationModule(Module):

    def __init__(self, transformations: list[Module]):
        super().__init__()
        self.transformations = transformations

    def forward(self, x: Tensor) -> Tensor:
        for transformation in self.transformations:
            x = transformation(x)
        return x


class Transformations(StrEnum):
    """
    Basic transformations that can be applied to a tensor.
    """

    def instance(self, config: ModelConfig, data_module: DataModule) -> Module:
        pass


def build_transformations(transformations: list[Transformations], config: ModelConfig, data_module: DataModule) -> Module:
    """
    Build a TransformationModule from a list of transformations.
    """
    return ConcatenatedTransformationModule([transformation.instance(config, data_module) for transformation in transformations])
