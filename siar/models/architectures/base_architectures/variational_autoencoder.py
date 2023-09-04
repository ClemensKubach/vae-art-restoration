import torch
from pythae.models import VAEConfig
from pythae.models import VAE as PythaeVAE
from pythae.models import VAE_LinNF as PythaeVAE_LinNF
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch.functional import F
from torch import nn, Tensor
from torch.nn import Module

from siar.models.architectures.base_architectures.sampler import GaussSampler, Sampler


class VAE(PythaeVAE):
    """Overriding the VAE class from pythae to allow not only self/input-reconstruction.
    And it allows custom samplers and a beta parameter for weighting the KL loss.
    """

    def __init__(
            self,
            arch_config: VAEConfig,
            encoder: BaseEncoder,
            decoder: BaseDecoder,
            sampler: Sampler | None = None,
            beta: float = 1
    ):
        super().__init__(model_config=arch_config, encoder=encoder, decoder=decoder)
        if sampler is None:
            self.sampler = GaussSampler()
        else:
            self.sampler = sampler
        self.beta = beta  # 0.01

    def forward(self, inputs: dict, **kwargs):
        """
        The VAE model

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        only_predict = False
        if 'labels' not in inputs:
            only_predict = True

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        assert mu.shape == log_var.shape
        assert len(mu.shape) == 2
        assert mu.shape[1] == self.model_config.latent_dim

        std = torch.exp(0.5 * log_var)
        z, eps = self.sampler(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        if only_predict:
            return ModelOutput(
                recon_x=recon_x,
                z=z,
                mu=mu,
                std=std
            )

        y = inputs["labels"]
        loss, recon_loss, kld = self.loss_function(recon_x, y, mu, log_var, z)

        return ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
            mu=mu,
            std=std
        )

    def loss_function(self, recon_x, x, mu, log_var, z):
        """From Pythae VAE loss function, but with beta parameter.
        By convention, mu and log_var, and z are expected to be the same shape of size (batch_size, latent_dim).
        """

        if self.model_config.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
        else:
            raise NotImplementedError("Only MSE and BCE losses are supported by the VAE for now")

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1) * self.beta

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


class VAE_LinNF(PythaeVAE_LinNF):
    """Fix for Pythae VAE_LinNF forward pass, because it does not support rhe reconstruction of non-input data"""

    def forward(self, inputs: dict, **kwargs):
        x = inputs["data"]
        only_predict = False
        if 'labels' not in inputs:
            only_predict = True

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)

        z0 = z

        log_abs_det_jac = torch.zeros((z0.shape[0],)).to(z.device)

        for layer in self.net:
            layer_output = layer(z)
            z = layer_output.out
            log_abs_det_jac += layer_output.log_abs_det_jac

        recon_x = self.decoder(z)["reconstruction"]

        if only_predict:
            return ModelOutput(
                recon_x=recon_x,
                z=z
            )

        y = inputs["labels"]
        loss, recon_loss, kld = self.loss_function(
            recon_x, y, mu, log_var, z0, z, log_abs_det_jac
        )

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )
        return output


class VdvaeArchitecture(VAE):

    def __init__(
            self,
            arch_config: VAEConfig,
            encoder: BaseEncoder,
            decoder: BaseDecoder
    ):
        super().__init__(arch_config=arch_config, encoder=encoder, decoder=decoder)

        if arch_config.reconstruction_loss != "mse":
            raise NotImplementedError("Only MSE loss is supported by the VDVAE for now")
        self.recon_loss_fn = nn.MSELoss(reduction="none")

    def forward(self, inputs: dict, **kwargs) -> ModelOutput:
        x = inputs["data"]
        only_predict = False
        if 'labels' not in inputs:
            only_predict = True

        encoder_output = self.encoder(x)
        skips_z: list[Tensor] = encoder_output["skips"]

        decoder_output = self.decoder(skips_z)
        recon_x = decoder_output.reconstruction
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
            skips_z=skips_z,
            z=decoder_output.z,
            mu=decoder_output.mu,
            std=decoder_output.std
        )

    def loss_function(self, recon_x, x, mu, log_var, z):
        recon_loss: torch.Tensor = self.recon_loss_fn(recon_x, x)
        dims_without_first = list(range(1, len(recon_loss.shape)))
        recon_loss = recon_loss.sum(dim=dims_without_first).mean(dim=0)
        kl_loss = self.decoder.kls.mean(dim=0)  # mean over batchsize
        loss = recon_loss + kl_loss
        return recon_loss, kl_loss, loss
