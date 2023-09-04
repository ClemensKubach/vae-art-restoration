from dataclasses import dataclass

from pythae.models import VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor, nn
from torch.nn import Module

from siar.models.architectures.base_architectures.sequential_variational_autoencoder import SeqVAE
from siar.models.architectures.base_architectures.variational_autoencoder import VAE
from siar.models.architectures.utils.components import ResBlock
from siar.models.modelmodule import ModelConfig, VAEModelModule


@dataclass
class SimpleResVaeConfig(VAEConfig):
    input_dim: tuple[int, int, int] = (3, 256, 256)
    latent_dim: int = 2000

    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_loss: str = "mse"
        self.uses_default_encoder: bool = False
        self.uses_default_decoder: bool = False


class SimpleResVaeEncoder(BaseEncoder):

    def __init__(self, config: VAEConfig):
        BaseEncoder.__init__(self)

        self.num_input_channels = config.input_dim[0]
        self.latent_dim = config.latent_dim

        self.encoder_net = nn.Sequential(
            # 3, 256, 256
            nn.Conv2d(self.num_input_channels, 16, kernel_size=3, stride=2, padding=1),
            # 16. 128. 128
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            # 16. 64, 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # 32, 32, 32
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            # 32, 16, 16
            ResBlock(in_channels=32, out_channels=8),
            # 32, 16, 16
            ResBlock(in_channels=32, out_channels=8),
            # 32, 16, 16
        )

        self.embedding = nn.Linear(32 * 16 * 16, self.latent_dim)
        self.log_var = nn.Linear(32 * 16 * 16, self.latent_dim)

    def forward(self, x: Tensor):
        x = self.encoder_net(x)
        x = x.view(x.size(0), -1)
        return ModelOutput(
            embedding=self.embedding(x),
            log_covariance=self.log_var(x)
        )


class SimpleResVaeDecoder(BaseDecoder):

    def __init__(self, config: VAEConfig):
        BaseDecoder.__init__(self)
        self.num_input_channels = config.input_dim[0]
        self.latent_dim = config.latent_dim

        self.embedding = nn.Linear(self.latent_dim, 32 * 16 * 16)

        self.decoder_net = nn.Sequential(
            # 32, 16, 16
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            # 32, 16, 16
            ResBlock(in_channels=32, out_channels=8),
            # 32, 16, 16
            ResBlock(in_channels=32, out_channels=8),
            # 32, 16, 16
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 32, 32, 32
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 16, 64, 64
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 16, 128, 128
            nn.ConvTranspose2d(16, self.num_input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z: Tensor):
        x = self.embedding(z)
        x = x.view(x.size(0), 32, 16, 16)
        reconstruction = self.decoder_net(x)
        return ModelOutput(
            reconstruction=reconstruction
        )


class SimpleResVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        vae_config: SimpleResVaeConfig = config.arch_config
        architecture = VAE(
            arch_config=vae_config,
            encoder=SimpleResVaeEncoder(vae_config),
            decoder=SimpleResVaeDecoder(vae_config)
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )


class SimpleSeqResVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        vae_config: SimpleResVaeConfig = config.arch_config
        architecture = SeqVAE(
            arch_config=vae_config,
            encoder=SimpleResVaeEncoder(vae_config),
            decoder=SimpleResVaeDecoder(vae_config)
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )
