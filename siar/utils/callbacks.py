import math
from enum import auto

import torch
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger

from unitorchplate.runner.run import Run
from unitorchplate.utils.callbacks import Callbacks as BaseCallbacks


class PlottingCallback(Callback):

    """
        plots original and prediction on validation set, same images for every epoch (look at data, maybe unshuffel val dataset)
    """

    def __init__(self, run: Run, num_images_to_plot: int = 10):
        self.logger: WandbLogger = run.logger
        self.buffer = []
        self.last_state = 'train'
        self.num_images_to_plot = num_images_to_plot

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.last_state == 'train':
            self.buffer = []
            self.last_state = 'val'
        self._plotting("val", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, only_idx=0, num_images=self.num_images_to_plot)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.last_state == 'val':
            self.buffer = []
            self.last_state = 'train'
        self._plotting("train", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, only_idx=0, num_images=self.num_images_to_plot)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.last_state == 'train' or self.last_state == 'val':
            self.buffer = []
            self.last_state = 'test'
        self._plotting("test", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, only_idx=0, num_images=2*self.num_images_to_plot)

    def _plotting(self, state: str, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, only_idx: int = 0, num_images: int = 10):
        batch_size = batch[0].shape[0]
        needed_batches = int(math.ceil(num_images / batch_size))
        accepted_batch_idx = {
            only_idx + i: idx for idx, i in enumerate(range(needed_batches))
        }

        if batch_idx in accepted_batch_idx:
            x, y = batch
            # show only first image of sequence
            if len(x.shape) == 5:
                x = x[:, 0]
            y_hat = outputs["recon_x"]
            missing_images = num_images - accepted_batch_idx[batch_idx] * batch_size
            if missing_images < batch_size:
                x = x[:missing_images]
                y = y[:missing_images]
                y_hat = y_hat[:missing_images]

            tensors_to_plot = [x, y_hat, y] if torch.not_equal(x, y).any() else [x, y_hat]
            images = torch.cat(tensors_to_plot, 3)
            images = torch.clamp(images, 0, 1)
            for i in images:
                if len(self.buffer) < num_images:
                    self.buffer.append(i)

            if len(self.buffer) >= num_images:
                self.logger.log_image(
                    key=f"{state}_images",
                    images=self.buffer[:num_images],  # [i for i in images],
                    caption=[f"{state} after epoch ..." for _ in self.buffer[:num_images]]
                )
                self.buffer = []

            # plot latent space
            # plt = plot_dim_reduced_latent_space(outputs["z"], show=False)
            # self.logger.log_image(f"{state}_latent_space", [wandb.Image(plt)])


class Callbacks(BaseCallbacks):
    PLOTTING = auto()

    def instance(self, run: Run) -> Callback | None:
        match self:
            case Callbacks.PLOTTING:
                return PlottingCallback(run)
