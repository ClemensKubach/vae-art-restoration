from abc import abstractmethod, ABC
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Union, MutableMapping, IO, Optional

import torch
from lightning import LightningModule, LightningDataModule
from torch.nn import Module
from typing_extensions import Self

from unitorchplate.models.losses.loss_types import LossType
from unitorchplate.models.model_types import ModelTypes
from unitorchplate.models.optimizer.optimizer_types import OptimizerType
from unitorchplate.models.optimizer.scheduler_types import LrSchedulerType
from unitorchplate.utils.processing.basic_transformations import Transformations


@dataclass
class LrSchedulerConfig:
    factor: float = 0.7
    patience: int = 30
    min_lr: float = 1e-8


@dataclass
class ModelConfig:
    model_type: ModelTypes = ModelTypes.CUSTOM
    loss_type: LossType = LossType.IN_ARCHITECTURE
    optimizer_type: OptimizerType = OptimizerType.ADAM
    lr_scheduler_type: LrSchedulerType = LrSchedulerType.NONE
    learning_rate: float = 1e-3
    batch_size: int = 2
    image_shape: tuple[int, int] | None = None
    num_classes: int | None = None
    model_module: LightningModule | None = None
    loss_mode: str | None = None
    loss_module: Module | None = None
    monitor_metric: str = "val_loss"
    preprocessing: list[Transformations] | None = None
    optimization_objective: str = "min"
    lr_scheduler_config: LrSchedulerConfig | None = None


class ModelModule(LightningModule, ABC):
    """Abstract class for standardizing the training, val, test splits, data preparation and transforms and lowering
    boilerplate.
    It is specifically designed that its implementation separates Lightning strictly from Pytorch code.
    """

    def __init__(
            self,
            architecture: torch.nn.Module | None,
            loss: torch.nn.Module | None,
            config: ModelConfig,
            preprocessing: torch.nn.Module | None = None
    ):
        """Abstract class for model modules.

        All subclasses must be able to accept and default to None for all torch.nn.Module arguments.
        Necessary for optimal checkpointing.
        """
        super().__init__()
        self.save_hyperparameters()

        # necessary for lightning tuner
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.optimizer = None
        self.scheduler = None

        if loss is None:
            print("No loss function provided. Maybe it is integrated as part of the architecture.")
        self.loss_fn = loss
        self.architecture = architecture
        self.preprocessing = preprocessing
        self.__config = config

        # for later use
        self._outputs_train = []
        self._outputs_val = []
        self._outputs_test = []

        # self.save_hyperparameters(asdict(config))

    def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
        """Override of HyperparametersMixin._set_hparams to allow skipping saving nn.Module attributes as hp.
        They are already saved in checkpoint. """
        hp = self._to_hparams_dict(hp)

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            hp_without_nn_modules = {
                k: v for k, v in hp.items() if not isinstance(v, torch.nn.Module) and not isinstance(v, LightningDataModule)
            }
            self.hparams.update(hp_without_nn_modules)
        else:
            # TODO implement skip for other types
            self._hparams = hp

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: Union[str, IO],
            map_location: Any = None,
            hparams_file: Optional[str] = None,
            strict: bool = False,
            **kwargs: Any,
    ) -> Self:
        return super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)

    @property
    def config(self) -> ModelConfig:
        return self.__config

    def forward(self, x):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return self.architecture(x)

    @abstractmethod
    def shared_step(self, batch, stage):
        """
        inputs, labels = batch

        logits_mask = self.forward(inputs)

        loss = self.loss_fn(logits_mask, labels)

        pred_mask = logits_mask

        # calculate metrics
        # ...

        prefix = f"{stage}_"

        console_metrics = {
            f"{prefix}loss": loss.mean()
        }
        additional_metrics = {
        }
        for k, v in console_metrics.items():
            self.log(k, v, prog_bar=True, logger=True)
        for k, v in additional_metrics.items():
            self.log(k, v, prog_bar=False, logger=True)

        if stage == "val":
            # self.log("hp_metric", f1.mean())
            pass

        if stage == "train":
            return loss
        else:
            return pred_mask
        """
        pass

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, stage="train")
        return out

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, stage="val")
        return out

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, stage="test")
        return out

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        out = self.shared_step(batch, stage="predict")
        return out

    def configure_optimizers(self):
        self.optimizer = self.config.optimizer_type.instance(self)
        assert self.optimizer is not None
        self.scheduler = self.config.lr_scheduler_type.instance(self)
        if self.scheduler is not None:
            return {
                'optimizer': self.optimizer,  # The optimizer to use
                'scheduler': self.scheduler,
                'monitor': self.config.monitor_metric,  # Metric to monitor for the scheduler
                'interval': 'epoch',    # When to step the scheduler (epoch-based)
                'frequency': 1          # How often to step the scheduler (every epoch)
            }
        else:
            return self.optimizer


        # assert self.optimizer is not None
        # if self.scheduler is not None:
        #     return [self.optimizer], [self.scheduler]
        # else:
        #     return self.optimizer
