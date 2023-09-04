import logging
import os
from abc import abstractmethod, ABC
from enum import auto
from typing import Iterable, Callable, Any

import optuna
import torch
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import Logger
from optuna import Study
from strenum import StrEnum
from torch.utils.data import DataLoader

from unitorchplate.models.model_types import ModelTypes
from unitorchplate.runner.run import Run, TrainerConfig, OptunaTuneTrainerConfig
from unitorchplate.utils.constants import DATAFILES_DIR, MODEL_DIR


console_logger = logging.getLogger(__name__)


class TuningModes(StrEnum):
    ITERATIVE = auto()
    PARALLEL = auto()


class LoadingModes(StrEnum):
    CHECKPOINT = auto()
    PT_MODEL = auto()


class Runner(ABC):

    def __init__(
            self,
            data_dir: str = DATAFILES_DIR,
            project_name: str = "default-project",
            num_workers: int = os.cpu_count(),
            auto_save: bool = False,
            random_seed: int = 42,
    ):
        self.__data_dir = data_dir
        self.__project_name = project_name
        self.auto_save = auto_save

        self.runs: dict[int, Run | None] = {}
        self.logger_template = self._def_logger_template()

        self.__num_workers = num_workers

        seed_everything(random_seed, workers=True)

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def project_name(self):
        return self.__project_name

    def get_logger(self, run: Run) -> Logger:
        return self.logger_template(name=run.name)

    @property
    def num_workers(self):
        return self.__num_workers

    def add_run(self, run: Run) -> int:
        run_id = len(self.runs)
        run.set_id(run_id)
        run.set_logger(self.get_logger(run))
        self.runs[run_id] = run
        print(f'Added Run {run.name} with following config: {run.config}')
        return run_id

    def prebuild_run(self, run: Run):
        return run.prebuild()

    def rebuild_run(self, run: Run, trainer_config: TrainerConfig):
        return run.rebuild(trainer_config=trainer_config)

    def del_run(self, run_id: int):
        self.runs[run_id] = None

    def get_last_valid_run(self) -> Run | None:
        last_valid_run_id = len(self.runs)
        run = None
        while run is None:
            last_valid_run_id -= 1
            if last_valid_run_id < 0:
                return None
            run = self.runs[last_valid_run_id]
        return run

    def sorted(self, metric: str = 'test_loss', reverse: bool = True):
        valid_runs = [r for r in self.runs.values() if r is not None and r.tested and metric in r.get_evaluation()]
        return sorted(valid_runs, key=lambda r: r.get_evaluation()[metric], reverse=reverse)

    def train(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None
    ):
        if trainer_config is None:
            trainer_config = TrainerConfig()

        run = self._check_for_simple_run(run, trainer_config)
        run.fit()
        if self.auto_save:
            self.save_model(run)

    def tune(
            self,
            mode: TuningModes = TuningModes.ITERATIVE,
            trainer_config: TrainerConfig | OptunaTuneTrainerConfig | None = None,
            runs: Iterable[Run] | None = None,
            optuna_objective: Callable = None
    ) -> list[dict[str, float]] | Study:
        if trainer_config is None:
            trainer_config = TrainerConfig(
                limit_train_batches=None,
                limit_val_batches=None,
                max_epochs=10
            )

        if mode == TuningModes.ITERATIVE:
            return self._iterative_tune(
                runs=runs,
                trainer_config=trainer_config,
            )
        elif mode == TuningModes.PARALLEL:
            assert isinstance(trainer_config, OptunaTuneTrainerConfig)
            return self._optuna_tune(
                objective=optuna_objective,
                trainer_config=trainer_config,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

    def test(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None
    ) -> list[dict[str, float]]:
        """Returns a list of dicts containing the metrics for each test run."""
        if trainer_config is None:
            trainer_config = TrainerConfig(
                limit_train_batches=None,
                limit_val_batches=None,
                max_epochs=1,
                devices=1,
                log_every_n_steps=1
            )
        run = self._check_for_test_run(run, trainer_config)
        return run.eval(ckpt_path=None)

    def predict(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None,
            data_loader: DataLoader | None = None
    ) -> list[torch.Tensor]:
        run = self._check_for_test_run(run, trainer_config)

        if data_loader is None:
            datamodule = run.data_module
        else:
            datamodule = None

        return run.trainer.predict(
            model=run.model,
            dataloaders=data_loader,
            datamodule=datamodule
        )

    def save_model(self, run: Run | None = None):
        if run is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if run.trained:
            torch.save(run.model, os.path.join(MODEL_DIR, f'model-{run.name}.pt'))
        else:
            raise ValueError("Model not trained yet. Please train it first.")

    def load_model(
            self,
            mode: LoadingModes = LoadingModes.CHECKPOINT,
            filename: str = None,
            path: str = None,
            model_type: ModelTypes | None = None,
            for_evaluation: bool = False
    ) -> LightningModule:
        if filename is not None:
            model_path = os.path.join(MODEL_DIR, filename)
        elif path is not None:
            model_path = path
        else:
            raise ValueError(f"Either name or path must be specified")

        if mode == LoadingModes.PT_MODEL:
            model = torch.load(model_path)
        elif mode == LoadingModes.CHECKPOINT:
            model = model_type.cls().load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown mode {mode}")
        if for_evaluation:
            model.eval()
        return model

    def close(self):
        print("Closing runner.")
        for rid in self.runs:
            run = self.runs[rid]
            run.logger.finalize(status="success")

    @abstractmethod
    def _def_logger_template(self) -> Any:
        pass

    def _check_for_simple_run(
            self,
            run: Run | None,
            trainer_config: TrainerConfig | None
    ) -> Run:
        if run is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")

        if not run.built:
            run.build(trainer_config=trainer_config)
        else:
            print("Run already built. Reusing existing build.")
        return run

    def _check_for_test_run(
            self,
            run: Run | None,
            trainer_config: TrainerConfig | None
    ) -> Run:
        if run is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")

        if not run.trained:
            raise ValueError("Run has not been trained yet. Please train first.")

        if not run.built:
            run.build(trainer_config=trainer_config)
        else:
            print("Run already built. Reusing existing build.")
        return run

    def _iterative_tune(self, runs: Iterable[Run] | None, trainer_config: TrainerConfig) -> list[dict[str, float]]:
        if runs is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")
            runs = [r for r in self.runs.values() if r is not None]  # [run]

        for run in runs:
            if not run.built:
                run.build(trainer_config=trainer_config)
            else:
                print(f"Run {run.name} already built. Reusing existing build.")

        metrics = []
        for run in runs:
            self.train(
                run=run,
                trainer_config=trainer_config
            )
            metrics.append(
                self.test(
                    run=run,
                    trainer_config=None
                )[0]
            )
        return metrics

    def _optuna_tune(self, objective: Callable, trainer_config: OptunaTuneTrainerConfig) -> Study:
        study = optuna.create_study(direction=trainer_config.direction)
        study.optimize(
            objective,
            n_trials=trainer_config.num_trials,
            timeout=trainer_config.timeout,
            n_jobs=trainer_config.num_jobs,
            show_progress_bar=trainer_config.show_progress_bar,
            gc_after_trial=trainer_config.gc_after_trial,
        )
        return study

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
