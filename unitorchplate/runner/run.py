import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from lightning import LightningDataModule, LightningModule, Trainer, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner import Tuner
from torch.nn import Module

from unitorchplate.data.datamodules import DataModuleConfig, DataModule
from unitorchplate.models.modelmodule import ModelConfig, ModelModule
from unitorchplate.utils.callbacks import Callbacks
from unitorchplate.utils.constants import CHECKPOINT_DIR
from unitorchplate.utils.processing.basic_transformations import build_transformations


@dataclass
class TrainerConfig:
    """Configuration for the Trainer, especially with Lightning parameters."""
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None
    max_epochs: int = 10
    devices: int | str = "auto"
    log_every_n_steps: int = 50
    monitor_metric: str = "val_loss"
    ckpt_every_n_epochs: int = 1
    ckpt_save_top_k: int = 1
    accumulate_grad_batches: int = 1
    precision: str = "32-true"


@dataclass
class OptunaTuneTrainerConfig(TrainerConfig):
    """Additional config parameters for the tuning with Optuna."""
    num_trials: int = None
    timeout: int = 60 * 60 * 2
    num_jobs: int = -1
    show_progress_bar: bool = True
    gc_after_trial: bool = True
    direction: str = "minimize"


@dataclass
class RunConfig:
    """Configuration for a run. It also supports specifying a custom Lightning trainer directly. Otherwise,
    it will be built from the TrainerConfig passed i.e. in run.train().
    """
    model_config: ModelConfig
    data_config: DataModuleConfig
    trainer: Trainer | None = None
    callbacks: list[Callbacks] | None = None


class Run:
    def __init__(
            self,
            config: RunConfig,
            name: str | None = None
    ):
        self.__id = None
        self.__name = name
        self.__config = config
        self.__built = False
        self.__trained = False
        self.__tested = False

        self.__data_module: DataModule | None = None
        self.__trainer: Trainer | None = config.trainer
        self.__tuner: Tuner | None = None
        self.__model: LightningModule | None = config.model_config.model_module
        self.__loss: Module | None = config.model_config.loss_module

        self.__evaluation: list[dict[str, Any]] = []

        self.__logger = None

    @property
    def id(self) -> int:
        return self.__id

    def set_id(self, id: int):
        self.__id = id

    @property
    def logger(self) -> Logger | None:
        return self.__logger

    def set_logger(self, logger: Logger):
        self.__logger = logger

    @property
    def name(self) -> str:
        return self.__name

    @property
    def config(self) -> RunConfig:
        return self.__config

    @property
    def data_module(self) -> DataModule | None:
        return self.__data_module

    @property
    def trainer(self) -> Trainer | None:
        return self.__trainer

    @property
    def model(self) -> LightningModule | None:
        return self.__model

    @model.setter
    def model(self, model: LightningModule):
        self.__model = model

    @property
    def loss(self) -> Module | None:
        return self.__loss

    @property
    def built(self) -> bool:
        return self.__built

    @property
    def trained(self) -> bool:
        return self.__trained

    @property
    def tested(self) -> bool:
        return self.__tested

    def get_evaluation(self, idx: int | None = None):
        if idx is None:
            return self.__evaluation[-1]
        if idx < 0:
            return self.__evaluation
        return self.__evaluation[idx]

    def build(self, trainer_config: TrainerConfig):
        """Before a task on a run can be executed, it has to be built. This means that the data module, the model, the
        loss and the trainer are built. If the trainer is not provided, it will be built from the trainer config.
        """
        self._build_pipeline(trainer_config=trainer_config)

    def prebuild(self):
        self._build_pipeline(prebuild=True)

    def fit(self):
        try:
            self.trainer.fit(
                model=self.model,
                datamodule=self.data_module
            )
            self.__trained = True
        except Exception as e:
            print(e)
            print(traceback.print_exception(e))

    def eval(self, ckpt_path: str | None = None) -> list[dict[str, Any]]:
        result = self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
            ckpt_path=ckpt_path
        )
        self.__evaluation.append(*result)
        self.__tested = True
        return result

    def rebuild(
            self,
            rebuild_data_module: bool = False,
            rebuild_trainer: bool = True,
            rebuild_model: bool = True,
            rebuild_loss: bool = False,
            trainer_config: TrainerConfig | None = None
    ):
        self.__built = False
        self._build_pipeline(
            force_build_data_module=rebuild_data_module,
            force_build_trainer=rebuild_trainer,
            force_build_model=rebuild_model,
            force_build_loss=rebuild_loss,
            trainer_config=trainer_config
        )

    def _build_pipeline(
            self,
            prebuild: bool = False,
            force_build_data_module: bool = False,
            force_build_trainer: bool = False,
            force_build_model: bool = False,
            force_build_loss: bool = False,
            trainer_config: TrainerConfig | None = None
    ):
        if self.built:
            print(f"Run {self.name} already built. Try to rebuild.")
            return

        if self.data_module is None or force_build_data_module:
            self.__data_module = self._build_datamodule()

        if self.config.model_config is None:
            raise ValueError("Model config is None")
        else:
            if self.name is None:
                self.__name = f'run-{self.config.model_config.model_type}-{str(datetime.now())}'
            if self.loss is None or force_build_loss:
                self.__loss = self._build_loss()
            if self.model is None or force_build_model:
                self.__model = self._build_model()

        if not prebuild:
            if self.trainer is None or force_build_trainer:
                if trainer_config is None:
                    raise ValueError("Trainer config is None")
                callbacks = self._build_callbacks(trainer_config)
                self.__trainer = self._build_trainer(trainer_config, callbacks)

            if self.__tuner is None:
                self.__tuner = self._build_tuner(self.__trainer)

            self.__built = True

    def _build_loss(self) -> Module:
        return self.config.model_config.loss_type.instance

    def _build_model(self) -> ModelModule:
        preprocessing_list = self.config.model_config.preprocessing if self.config.model_config.preprocessing is not None else []
        preprocessing_module = build_transformations(preprocessing_list, self.config.model_config, self.data_module)
        return self.config.model_config.model_type.instance(self.config.model_config, preprocessing_module)

    def _build_datamodule(self) -> LightningDataModule:
        # transform = get_transforms(self.config.augmentation)
        return self.config.data_config.instance()

    def _build_callbacks(self, config: TrainerConfig) -> list[Callback]:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpointing_callback = ModelCheckpoint(
            # monitor=f'{self.id}/val_f1',
            monitor=config.monitor_metric,
            save_top_k=config.ckpt_save_top_k,
            auto_insert_metric_name=True,
            every_n_epochs=config.ckpt_every_n_epochs,
        )
        additional_callbacks = self.config.callbacks
        if additional_callbacks is None:
            additional_callbacks = []
        callbacks = [
            lr_monitor,
            checkpointing_callback,
            *[c.instance(self) for c in additional_callbacks]
        ]
        return callbacks

    def _build_trainer(self, config: TrainerConfig, callbacks: list[Callback]) -> Trainer:
        if self.logger is None:
            print("No logger provided!")
        trainer = Trainer(
            max_epochs=config.max_epochs,
            logger=self.logger,
            num_nodes=-1,
            limit_train_batches=config.limit_train_batches,
            limit_val_batches=config.limit_val_batches,
            enable_checkpointing=True,
            callbacks=callbacks,
            default_root_dir=CHECKPOINT_DIR,
            devices=config.devices,
            log_every_n_steps=config.log_every_n_steps,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=config.accumulate_grad_batches,
            precision=config.precision
        )
        return trainer

    def _build_tuner(self, trainer: Trainer):
        tuner = Tuner(trainer)
        if self.config.model_config.batch_size is None:
            tuner.scale_batch_size(
                model=self.model,
                mode="power",
                datamodule=self.data_module
            )
        if self.config.model_config.learning_rate is None:
            tuner.lr_find(
                model=self.model,
                datamodule=self.data_module
            )
        return tuner
