from dataclasses import dataclass

from pythae.models import VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor, nn
from torch.nn import Module

from siar.models.architectures.base_architectures.variational_autoencoder import VAE
from siar.models.architectures.utils.components import TripleRes
from siar.models.modelmodule import ModelConfig, VAEModelModule


@dataclass
class ResVaeConfig(VAEConfig):
    input_dim: tuple[int, int, int] = (3, 256, 256)
    latent_dim: int = 2000

    def __post_init__(self):
        super().__post_init__()
        self.reconstruction_loss: str = "mse"
        self.uses_default_encoder: bool = False
        self.uses_default_decoder: bool = False


class ResVaeEncoder(BaseEncoder):

    def __init__(self, config: VAEConfig):
        BaseEncoder.__init__(self)

        self.num_input_channels = config.input_dim[0]
        self.latent_dim = config.latent_dim

        self.encoder_net = nn.Sequential(
            # 3, 256, 256
            nn.Conv2d(self.num_input_channels, 128, kernel_size=3, stride=2, padding=1),
            # 128. 128. 128
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128. 128. 128
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            # 128. 64, 64
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128.64. 64
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            # 128. 32, 32
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128. 32. 
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            # 128. 16, 16
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128. 16. 16
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            # 128. 8, 8
            
        )

        self.embedding = nn.Linear(128 * 8 * 8, self.latent_dim)
        self.log_var = nn.Linear(128 * 8 * 8, self.latent_dim)

    def forward(self, x: Tensor):
        x = self.encoder_net(x)
        x = x.view(x.size(0), -1)
        return ModelOutput(
            embedding=self.embedding(x),
            log_covariance=self.log_var(x)
        )


class ResVaeDecoder(BaseDecoder):

    def __init__(self, config: VAEConfig):
        BaseDecoder.__init__(self)
        self.num_input_channels = config.input_dim[0]
        self.latent_dim = config.latent_dim

        self.embedding = nn.Linear(self.latent_dim,128 * 8 * 8)

        self.decoder_net = nn.Sequential(
            # 128, 8, 8
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 128, 16, 16
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128, 16, 16
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 128, 32, 32
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128, 32, 32
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 128, 64, 64
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128, 64, 64
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 128, 128, 128
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            TripleRes(128, 32, 128),
            # 128, 128, 128
            nn.GELU(),
            nn.ConvTranspose2d(128, self.num_input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            # 3, 256, 256
        )

    def forward(self, z: Tensor):
        x = self.embedding(z)
        x = x.view(x.size(0), 128, 8, 8)
        reconstruction = self.decoder_net(x)
        return ModelOutput(
            reconstruction=reconstruction
        )


class ResVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        vae_config: ResVaeConfig = config.arch_config
        architecture = VAE(
            arch_config=vae_config,
            encoder=ResVaeEncoder(vae_config),
            decoder=ResVaeDecoder(vae_config)
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )
