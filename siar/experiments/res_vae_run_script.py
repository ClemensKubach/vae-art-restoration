from siar.data.datamodules import SiarDataModuleConfig
from siar.data.datasets import SiarDatasetConfig
from siar.models.architectures.res_vae import ResVaeConfig
from siar.models.model_types import ModelTypes
from siar.models.modelmodule import ModelConfig
from siar.runner.runner import SiarRunner, SiarRunConfig

from siar.utils.callbacks import Callbacks
from siar.utils.entities import Task
from unitorchplate.models.losses.loss_types import LossType
from unitorchplate.runner.run import Run, TrainerConfig


def main():
    print('Executing...')
    with SiarRunner(
        task=Task.SELF_RECONSTRUCTION,
        project_prefix='res-vae',
        auto_save=False
    ) as runner:
        batch_size: int = 2
        run = Run(
            name=None,
            config=SiarRunConfig(
                model_config=ModelConfig(
                    learning_rate=1e-4,
                    batch_size=batch_size,
                    model_type=ModelTypes.ResVAE,
                    loss_type=LossType.IN_ARCHITECTURE,
                    arch_config=ResVaeConfig(
                        latent_dim=2000,
                    )
                ),
                data_config=SiarDataModuleConfig(
                    dataset_config=SiarDatasetConfig(
                        return_sequence=False,
                        gt_target=True if runner.task == Task.ORIGINAL_RECONSTRUCTION else False,
                        self_target=True if runner.task == Task.SELF_RECONSTRUCTION else False
                    ),
                    batch_size=batch_size,
                ),
                callbacks=[Callbacks.PLOTTING]
            )
        )
        runner.add_run(run)

        runner.train(
            run=run,
            trainer_config=TrainerConfig(
                max_epochs=200,
                limit_train_batches=1000,
                limit_val_batches=100,
                log_every_n_steps=50,
                ckpt_every_n_epochs=50,
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
