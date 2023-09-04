from dataclasses import dataclass

from pythae.models import VAE_LinNF_Config
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor, nn
from torch.nn import Module
import torch

from siar.models.architectures.base_architectures.variational_autoencoder import VAE_LinNF
from siar.models.modelmodule import ModelConfig, VAEModelModule


@dataclass
class NfVaeConfig(VAE_LinNF_Config):
    input_dim: tuple[int, int, int] = (3, 256, 256)
    latent_dim: int = 2000
    reconstruction_loss: str = "mse"
    flows: list[str] | None = None

    def __post_init__(self):
        if self.flows is None:
            self.flows = ["Radial", "Radial", "Planar", "Planar", "Planar", "Radial", "Radial"]
        self.uses_default_encoder: bool = False
        self.uses_default_decoder: bool = False
        super().__post_init__()


class ResBlockNF(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class NfVaeEncoder(BaseEncoder):

    def __init__(self, config: NfVaeConfig):
        BaseEncoder.__init__(self)

        # init params
        self.latent_dim = config.latent_dim
        self.in_channel, self.img_h, self.img_w = config.input_dim  # (3,256,256)
        self.h = self.img_h // 32  # img size from 256 to 16,(2^4=32)
        self.w = self.img_w // 32
        hw = self.h * self.w
        self.hidden_dims = [128, 128, 64, 64, 32]

        layers = []
        for hidden_dim in self.hidden_dims[:2]:
            layers += [nn.Conv2d(self.in_channel, hidden_dim, 4, 2, 1),
                       # 3，2，1-->img size/2 ; 3,1,1 --> img size remain
                       ResBlockNF(hidden_dim, hidden_dim // 2),
                       ResBlockNF(hidden_dim, hidden_dim // 2),
                       ResBlockNF(hidden_dim, hidden_dim // 2)]
            self.in_channel = hidden_dim

        for hidden_dim in self.hidden_dims[2:]:
            layers += [nn.Conv2d(self.in_channel, hidden_dim, 3, 2, 1),
                       ResBlockNF(hidden_dim, hidden_dim // 2),
                       ResBlockNF(hidden_dim, hidden_dim // 2),
                       ResBlockNF(hidden_dim, hidden_dim // 2)]
            self.in_channel = hidden_dim

        self.encoder_net = nn.Sequential(*layers)
        self.embedding = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)

    def forward(self, x: Tensor):
        x = self.encoder_net(x)
        x = x.view(x.size(0), -1)
        return ModelOutput(
            embedding=self.embedding(x),
            log_covariance=self.log_var(x)
        )


class NfVaeDecoder(BaseDecoder):

    def __init__(self, config: NfVaeConfig):
        BaseDecoder.__init__(self)
        # init params
        self.latent_dim = config.latent_dim
        self.in_channel, self.img_h, self.img_w = config.input_dim  # (3,256,256)
        self.h = self.img_h // 32  # img size from 256 to 16,(2^4=32)
        self.w = self.img_w // 32
        hw = self.h * self.w
        self.hidden_dims = [128, 128, 64, 64, 32]

        # decoder
        layers = []
        self.embedding = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            layers += [nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1], 3, 2, 1, 1),
                       ResBlockNF(self.hidden_dims[i + 1], self.hidden_dims[i + 1] // 2),
                       ResBlockNF(self.hidden_dims[i + 1], self.hidden_dims[i + 1] // 2),
                       ResBlockNF(self.hidden_dims[i + 1], self.hidden_dims[i + 1] // 2)]
        layers += [nn.ConvTranspose2d(self.hidden_dims[-1], 3, 3, 2, 1, 1),
                   nn.Tanh()]
        self.decoder_net = nn.Sequential(*layers)

    def forward(self, z: Tensor):
        x = self.embedding(z)
        x = x.view(-1, self.hidden_dims[0], self.h, self.w)
        reconstruction = self.decoder_net(x)
        return ModelOutput(
            reconstruction=reconstruction
        )


class NfVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        vae_config: NfVaeConfig = config.arch_config
        architecture = VAE_LinNF(
            model_config=vae_config,
            encoder=NfVaeEncoder(vae_config),
            decoder=NfVaeDecoder(vae_config),
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )
