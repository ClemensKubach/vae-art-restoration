from dataclasses import dataclass

from lightning import LightningModule
from pythae.config import BaseConfig
from pythae.models import VAE, VAEConfig, BaseAE, BaseAEConfig, VAE_LinNF, VAE_LinNF_Config
from pythae.models.base.base_utils import ModelOutput
from torch import Tensor
from torch.nn import Module

from siar.models.model_types import ModelTypes
from unitorchplate.models.losses.loss_types import LossType
from unitorchplate.models.modelmodule import ModelModule
from unitorchplate.models.modelmodule import ModelConfig as BaseModelConfig
from unitorchplate.models.optimizer.optimizer_types import OptimizerType
from unitorchplate.models.optimizer.scheduler_types import LrSchedulerType
from unitorchplate.utils.processing.basic_transformations import Transformations


@dataclass
class ModelConfig(BaseModelConfig):
    arch_config: BaseConfig | None = None
    model_type: ModelTypes = ModelTypes.SimpleResVAE
    loss_type: LossType = LossType.IN_ARCHITECTURE
    optimizer_type: OptimizerType = OptimizerType.ADAM
    lr_scheduler_type: LrSchedulerType = LrSchedulerType.NONE
    learning_rate: float = 1e-3
    batch_size: int = 32
    image_shape: tuple[int, int] | None = (256, 256)
    num_classes: int | None = None
    model_module: LightningModule | None = None
    loss_mode: str | None = None
    loss_module: Module | None = None
    monitor_metric: str = "val_recon_loss"
    preprocessing: list[Transformations] | None = None


class VAEModelModule(ModelModule):
    """LightningModule for VAE models compatible with the pythae library."""

    def __init__(
            self,
            architecture: BaseAE,
            config: ModelConfig,
            preprocessing: Module | None = None
    ):
        super().__init__(architecture, None, config, preprocessing)

    def forward(self, x: dict | Tensor) -> ModelOutput:
        if isinstance(x, Tensor):
            x = {
                'data': x
            }
        return super().forward(x)

    def shared_step(self, batch, stage):
        inputs, labels = batch

        pythae_input = {
            'data': inputs,
            'labels': labels
        }

        model_output = self.forward(pythae_input)

        loss_keys = ["loss", "recon_loss", "reg_loss"]

        prefix = f"{stage}_"

        console_metrics = {
            f"{prefix}{loss_key}": model_output[loss_key] for loss_key in loss_keys
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

        # if stage == "train":
        #     return model_output["loss"]
        # else:
        #     return {k: model_output[k] for k in ("recon_x", "z")}
        return model_output
