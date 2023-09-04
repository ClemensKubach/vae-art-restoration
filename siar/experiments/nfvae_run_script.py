from siar.data.datamodules import SiarDataModuleConfig
from siar.data.datasets import SiarDatasetConfig
from siar.models.architectures.nf_vae import NfVaeConfig
from siar.models.model_types import ModelTypes
from siar.models.modelmodule import ModelConfig
from siar.runner.runner import SiarRunner, SiarRunConfig

from siar.utils.callbacks import Callbacks
from siar.utils.entities import Task
from unitorchplate.models.losses.loss_types import LossType
from unitorchplate.models.modelmodule import LrSchedulerConfig
from unitorchplate.models.optimizer.optimizer_types import OptimizerType
from unitorchplate.models.optimizer.scheduler_types import LrSchedulerType
from unitorchplate.runner.run import Run, TrainerConfig


def main():
    print('Executing...')
    with SiarRunner(
        task=Task.SELF_RECONSTRUCTION,
        project_prefix='nf-vae',
        auto_save=False
    ) as runner:
        batch_size = 16
        run = Run(
            name=None,
            config=SiarRunConfig(
                model_config=ModelConfig(
                    learning_rate=1e-3,
                    batch_size=batch_size,
                    model_type=ModelTypes.NF_VAE,
                    loss_type=LossType.IN_ARCHITECTURE,
                    arch_config=NfVaeConfig(),
                    optimizer_type=OptimizerType.ADAM,
                    lr_scheduler_type=LrSchedulerType.NONE,
                    lr_scheduler_config=LrSchedulerConfig()
                ),
                data_config=SiarDataModuleConfig(
                    dataset_config=SiarDatasetConfig(
                        return_sequence=False,
                        gt_target=True if runner.task == Task.ORIGINAL_RECONSTRUCTION else False,
                        self_target=True if runner.task == Task.SELF_RECONSTRUCTION else False
                    ),
                    batch_size=batch_size,
                    num_workers=1,
                ),
                callbacks=[Callbacks.PLOTTING]
            )
        )
        runner.add_run(run)

        runner.train(
            run=run,
            trainer_config=TrainerConfig(
                max_epochs=100,
                limit_train_batches=1000,
                limit_val_batches=100,
                log_every_n_steps=50,
                ckpt_every_n_epochs=5,
                ckpt_save_top_k=-1
            )
        )

        result_dict = runner.test(
            run=run,
            trainer_config=None,
        )[0]
        print(result_dict)
        print('Done.')


if __name__ == '__main__':
    main()
