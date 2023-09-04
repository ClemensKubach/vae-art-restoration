from siar.data.datamodules import SiarDataModuleConfig
from siar.data.datasets import SiarDatasetConfig
from siar.models.architectures.simple_res_vae import SimpleResVaeConfig
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
        task=Task.ORIGINAL_RECONSTRUCTION,
        project_prefix='simple-run',
        auto_save=False
    ) as runner:
        batch_size = 32
        run = Run(
            name=None,
            config=SiarRunConfig(
                model_config=ModelConfig(
                    learning_rate=1e-4,
                    batch_size=batch_size,
                    model_type=ModelTypes.SimpleResVAE,
                    loss_type=LossType.IN_ARCHITECTURE,
                    arch_config=SimpleResVaeConfig(
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
        # If you want to load a model for further training, uncomment the following lines and change the path.
        # run.model = runner.load_model(
        #     path=f'{LOGGING_DIR}/simple-res-vae-original_reconstruction/omgduza9/checkpoints/epoch=99-step=100000.ckpt',
        #     model_type=ModelTypes.SimpleResVAE,
        #     for_evaluation=False
        # )
        runner.add_run(run)

        runner.train(
            run=run,
            trainer_config=TrainerConfig(
                max_epochs=1,
                limit_train_batches=None,
                limit_val_batches=None,
                log_every_n_steps=50,
                ckpt_every_n_epochs=10,
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
