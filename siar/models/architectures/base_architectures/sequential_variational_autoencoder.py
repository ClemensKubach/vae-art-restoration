import torch
from pythae.models.base.base_utils import ModelOutput
from torch import Tensor

from siar.models.architectures.base_architectures.variational_autoencoder import VAE, VdvaeArchitecture


class SeqVAE(VAE):
    """Overriding the VAE class from pythae to allow not only self/input-reconstruction."""
    def forward(self, inputs: dict, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        seq_x = inputs["data"]  # (batch_size, seq_len, channels, height, width)
        only_predict = False
        if 'labels' not in inputs:
            only_predict = True

        mu_sum = torch.zeros((seq_x.shape[0], self.model_config.latent_dim), device=seq_x.device)
        log_var_sum = torch.zeros((seq_x.shape[0], self.model_config.latent_dim), device=seq_x.device)
        z_sum = torch.zeros((seq_x.shape[0], self.model_config.latent_dim), device=seq_x.device)
        for seq_idx in range(seq_x.shape[1]):
            x = seq_x[:, seq_idx, ...]
            encoder_output = self.encoder(x)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            mu_sum += mu
            log_var_sum += log_var

            z, eps = self._sample_gauss(mu, std)
            z_sum += z
        z = z_sum / seq_x.shape[1]

        recon_x = self.decoder(z)["reconstruction"]

        if only_predict:
            return ModelOutput(
                recon_x=recon_x,
                z=z
            )

        y = inputs["labels"]
        mu_avg = mu_sum / seq_x.shape[1]
        log_var_avg = log_var_sum / seq_x.shape[1]
        loss, recon_loss, kld = self.loss_function(recon_x, y, mu_avg, log_var_avg, z)

        return ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )


class SeqVdvaeArchitecture(VdvaeArchitecture):

    def forward(self, inputs: dict, **kwargs) -> ModelOutput:
        seq_x = inputs["data"]
        only_predict = False
        if 'labels' not in inputs:
            only_predict = True

        skips_z_sum: list[Tensor] | None = None
        for seq_idx in range(seq_x.shape[1]):
            x = seq_x[:, seq_idx, ...]
            encoder_output = self.encoder(x)
            skips: list[Tensor] = encoder_output['skips']
            if skips_z_sum is None:
                skips_z_sum = [z.detach().clone() for z in skips]
            else:
                for z, z_sum in zip(encoder_output['skips'], skips_z_sum):
                    z_sum += z
        skips_z = [z_sum / seq_x.shape[1] for z_sum in skips_z_sum]
        # skips_z are not the latent vectors z directly, but the intermediate representation between encoder and
        # decoder from which in the decoder the latent vectors z are sampled.
        decoder_output = self.decoder(skips_z)
        recon_x = decoder_output["reconstruction"]

        if only_predict:
            return ModelOutput(
                recon_x=recon_x,
                skips_z=skips_z,
                z=decoder_output.z,
                mu=decoder_output.mu,
                std=decoder_output.std
            )

        y = inputs["labels"]
        recon_loss, kl_loss, loss = self.loss_function(
            recon_x, y,
            None, None, None  # handled in decoder
        )
        return ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kl_loss,
            loss=loss,
            recon_x=recon_x,
            pre_z=skips_z,
            z=decoder_output.z,
            mu=decoder_output.mu,
            std=decoder_output.std
        )
