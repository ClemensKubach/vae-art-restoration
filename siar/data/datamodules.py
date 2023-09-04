import os
from dataclasses import dataclass

from siar.data.datasets import SiarDatasetConfig
from unitorchplate.data.datamodules import DataModule, DataModuleConfig


@dataclass
class SiarDataModuleConfig(DataModuleConfig):
    dataset_config: SiarDatasetConfig = SiarDatasetConfig()
    batch_size: int = 32
    shuffle_train: bool = True
    train_size: float = 0.8
    val_size: float = 0.1
    num_workers: int = os.cpu_count()
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2

    def instance(self) -> DataModule:
        return SiarDataModule(self)


class SiarDataModule(DataModule):
    def __init__(
            self,
            config: SiarDataModuleConfig
    ):
        super().__init__(config)
        