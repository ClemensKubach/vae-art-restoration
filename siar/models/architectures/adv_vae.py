from dataclasses import dataclass

from pythae.models import VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor, nn
from torch.nn import Module
import torch

from siar.models.architectures.base_architectures.sequential_variational_autoencoder import SeqVAE
from siar.models.architectures.base_architectures.variational_autoencoder import VAE
from siar.models.architectures.utils.components import ResBlock
from siar.models.modelmodule import ModelConfig, VAEModelModule


@dataclass
class AdvVaeConfig(VAEConfig):
    input_dim: tuple[int, int, int] = (3, 256, 256)
    hidden_size: int = 32
    hidden_reduced_size: int = 8
    num_layers: int = 4
    num_blocks_per_layer: int = 3
    latent_depth: int = 16

    # run # 2 - prime plant
    #	self.hidden_size = 32
    #	self.hidden_reduced_size = 8
    #	self.num_layers = 5
    #	self.num_blocks_per_layer = 2

    # run # 3 -
    #	self.hidden_size = 64
    #	self.hidden_reduced_size = 16
    #	self.num_layers = 3
    #	self.num_blocks_per_layer = 3

    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_loss: str = "mse"
        self.uses_default_encoder: bool = False
        self.uses_default_decoder: bool = False
        size_x = self.input_dim[1] // 2**(self.num_layers-1)
        size_y = self.input_dim[2] // 2**(self.num_layers-1)
        self.latent_dim: int = self.latent_depth * size_x * size_y


class AdvVaeEncoder(BaseEncoder):

    def __init__(self, config: AdvVaeConfig):
        BaseEncoder.__init__(self)
        self.config = config
        self.num_input_channels = config.input_dim[0]

        layers = [nn.Conv2d(self.num_input_channels, self.config.hidden_size, kernel_size=3, stride=1, padding=1)]

        for i in range(self.config.num_layers):  # use config instead of number
            for j in range(self.config.num_blocks_per_layer):
                layers.append(ResBlock(self.config.hidden_size, self.config.hidden_reduced_size))
            if i != self.config.num_layers - 1:  # only reduce size if not last activation
                layers.append(torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1))  # reduce w/h by half

        self.encoder_net = torch.nn.Sequential(*layers)

        self.final_layer = nn.Conv2d(self.config.hidden_size, 2 * self.config.latent_depth, kernel_size=3, stride=1,
                                     padding=1)

    def forward(self, x: Tensor):
        x = self.encoder_net(x)
        x = self.final_layer(x)
        return ModelOutput(
            embedding=torch.reshape(x[:, :self.config.latent_depth], (x.shape[0], -1)),
            log_covariance=torch.reshape(x[:, self.config.latent_depth:], (x.shape[0], -1))
        )


class AdvVaeDecoder(BaseDecoder):

    def __init__(self, config: AdvVaeConfig):
        BaseDecoder.__init__(self)
        self.config = config
        self.num_input_channels = config.input_dim[0]

        layers = [nn.Conv2d(self.config.hidden_size, self.num_input_channels, kernel_size=3, stride=1, padding=1)]

        for i in range(self.config.num_layers):  # use config instead of number
            for j in range(self.config.num_blocks_per_layer):
                layers.append(ResBlock(self.config.hidden_size, self.config.hidden_reduced_size))
            if i != self.config.num_layers - 1:  # only reduce size if not last activation
                layers.append(torch.nn.UpsamplingNearest2d(scale_factor=2))  # reduce w/h by half
        layers.append(nn.Conv2d(self.config.latent_depth, self.config.hidden_size, kernel_size=3, stride=1,
        padding=1))

        self.decoder_net = torch.nn.Sequential(*reversed(layers))

    def forward(self, z: Tensor):
        size_x = self.config.input_dim[1] // 2**(self.config.num_layers-1)
        size_y = self.config.input_dim[2] // 2**(self.config.num_layers-1)

        z = torch.reshape(z, (z.shape[0], self.config.latent_depth, size_x, size_y))
        reconstruction = self.decoder_net(z)
        return ModelOutput(
            reconstruction=reconstruction
        )


class AdvVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        vae_config: AdvVaeConfig = config.arch_config
        architecture = VAE(
            arch_config=vae_config,
            encoder=AdvVaeEncoder(vae_config),
            decoder=AdvVaeDecoder(vae_config),
            beta=0.01
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )

    def reconstruct_from_single_latent(self, z):
        recon_x = self.architecture.decoder(z[None, :])["reconstruction"]
        return recon_x


class SeqAdvVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        vae_config: AdvVaeConfig = config.arch_config
        architecture = SeqVAE(
            arch_config=vae_config,
            encoder=AdvVaeEncoder(vae_config),
            decoder=AdvVaeDecoder(vae_config)
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )
