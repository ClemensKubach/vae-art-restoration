import functools
import os
from dataclasses import dataclass
from typing import Any

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from siar.data.datamodules import SiarDataModuleConfig
from siar.models.modelmodule import ModelConfig
from siar.utils.callbacks import Callbacks
from siar.utils.entities import Task
from unitorchplate.runner.run import RunConfig
from unitorchplate.runner.runner import Runner
from unitorchplate.utils.constants import LOGGING_DIR, DATAFILES_DIR


@dataclass
class SiarRunConfig(RunConfig):
    model_config: ModelConfig = ModelConfig()
    data_config: SiarDataModuleConfig = SiarDataModuleConfig()
    trainer: Trainer | None = None
    callbacks: list[Callbacks] | None = None


class SiarRunner(Runner):

    def __init__(
            self,
            task: Task = Task.SELF_RECONSTRUCTION,
            data_dir: str = DATAFILES_DIR,
            project_prefix: str | None = None,
            num_workers: int = os.cpu_count(),
            auto_save: bool = False,
    ):
        self.task = task
        if project_prefix is None:
            project_prefix = "default-siar-project"
        project_name = f"{project_prefix}-{task.value}"
        super().__init__(data_dir, project_name, num_workers, auto_save)

    def _def_logger_template(self) -> Any:
        if not os.path.exists(LOGGING_DIR):
            os.makedirs(LOGGING_DIR)
        return functools.partial(
            WandbLogger,
            project=self.project_name,
            dir=LOGGING_DIR,
            log_model=True,
            save_dir=LOGGING_DIR,
            anonymous=True,
        )
