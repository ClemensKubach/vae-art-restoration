from dataclasses import dataclass

import torch
import torch.nn as nn
from pythae.models import VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor
from torch.nn import Module

from siar.models.architectures.base_architectures.sequential_variational_autoencoder import SeqVdvaeArchitecture
from siar.models.architectures.base_architectures.variational_autoencoder import VdvaeArchitecture
from siar.models.architectures.utils.components import ResBlockVDVAE, TripleRes
from siar.models.architectures.utils.funcs import draw_gaussian_diag_samples, gaussian_analytical_kl
from siar.models.modelmodule import ModelConfig, VAEModelModule


@dataclass
class VdvaeConfig(VAEConfig):
    input_dim: tuple[int, int, int] = (3, 256, 256)
    resBlock_dim_in: int = 128
    resBlock_dim_mid: int = 32
    num_blocks: int = 5
    skips_bool: list[bool] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.skips_bool is None:
            self.skips_bool = [True] * self.num_blocks
        self.latent_dim: int = 0  # unused
        self.reconstruction_loss: str = "mse"
        self.uses_default_encoder: bool = False
        self.uses_default_decoder: bool = False


class TopDown(Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels

        self.resBlockQ = ResBlockVDVAE(in_channels * 2, hid_channels, out_channels * 2)
        self.resBlockP = ResBlockVDVAE(in_channels, hid_channels, out_channels * 2 + in_channels)
        self.conv11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                      padding=0)
        # self.block = tripleRes(in_channels, hid_channels, out_channels)
        self.block = ResBlockVDVAE(in_channels, hid_channels, out_channels, residual=True)

    def sample(self, x, skip):
        qm, qv = self.resBlockQ(torch.cat([x, skip], dim=1)).chunk(2, dim=1)
        p = self.resBlockP(x)
        pm, pv, xpp = p[:, :self.out_channels, ...], p[:, self.out_channels:self.out_channels * 2, ...], p[:,
                                                                                                         self.out_channels * 2:,
                                                                                                         ...]
        x = x + xpp

        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl, qm, qv

    def sample_noskip(self, x):
        p = self.resBlockP(x)
        pm, pv, xpp = p[:, :self.out_channels, ...], p[:, self.out_channels:self.out_channels * 2, ...], p[:,
                                                                                                         self.out_channels * 2:,
                                                                                                         ...]
        x = x + xpp

        z = draw_gaussian_diag_samples(pm, pv)

        return z, x, pm, pv

    def forward(self, x, skip, first: bool = False) -> ModelOutput:
        kls = torch.zeros((x.shape[0]), device=x.device)
        n = 3
        if first:
            n = 4
        z_list = []
        mu_list = []
        std_list = []
        if torch.is_tensor(skip):
            for _ in range(n):
                z, x, kl, mu, log_covar = self.sample(x, skip)
                z_list.append(z.detach().cpu())
                mu_list.append(mu.detach().cpu())
                std_list.append(torch.exp(log_covar).detach().cpu())

                kls += kl
                x = x + self.conv11(z)
                x = self.block(x)
        else:
            for _ in range(n):
                z, x, mu, log_covar = self.sample_noskip(x)
                z_list.append(z.detach().cpu())
                mu_list.append(mu.detach().cpu())
                std_list.append(torch.exp(log_covar).detach().cpu())

                x = x + self.conv11(z)
                x = self.block(x)
        return ModelOutput(
            x=x,
            kls=kls,
            z=z_list,
            mu=mu_list,
            std=std_list
        )


class VdVaeEncoder(BaseEncoder):
    def __init__(self, config: VdvaeConfig):
        BaseEncoder.__init__(self)

        self.num_input_channels = config.input_dim[0]
        num_input_channels_res = config.resBlock_dim_in
        num_mid_channels_res = config.resBlock_dim_mid
        self.num_blocks = config.num_blocks
        self.skips_bool = config.skips_bool

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(self.num_input_channels, num_input_channels_res, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([
            TripleRes(num_input_channels_res, num_mid_channels_res, num_input_channels_res) for _ in
            range(self.num_blocks)
        ])
        self.endblock = nn.Sequential(
            TripleRes(num_input_channels_res, num_mid_channels_res, num_input_channels_res),
            ResBlockVDVAE(num_input_channels_res, num_mid_channels_res, num_input_channels_res, residual=True),
        )

    def forward(self, x: Tensor):
        skips = []
        x = self.conv(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.skips_bool[i]:
                skips.append(x)
            else:
                skips.append(None)
            x = self.pool(x)

        # last layer - no pooling
        x = self.endblock(x)
        skips.append(x)
        return ModelOutput(skips=skips)


class VdVaeDecoder(BaseDecoder):
    def __init__(self, config: VdvaeConfig):
        BaseDecoder.__init__(self)
        self.num_input_channels = config.input_dim[0]
        num_input_channels_res = config.resBlock_dim_in
        num_mid_channels_res = config.resBlock_dim_mid
        self.num_blocks = config.num_blocks

        self.kls: Tensor | None = None

        self.blocks = torch.nn.ModuleList(
            [TopDown(num_input_channels_res, num_mid_channels_res, num_input_channels_res) for _ in
             range(self.num_blocks)])
        self.endblock = TopDown(num_input_channels_res, num_mid_channels_res, num_input_channels_res)
        self.unpool = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = torch.nn.Conv2d(num_input_channels_res, self.num_input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, skips: list[Tensor]):
        kls = torch.zeros((skips[-1].shape[0]), device=skips[-1].device)
        first = True
        x = skips[-1]
        nested_z_list = []
        nested_mu_list = []
        nested_std_list = []
        for i, block in enumerate(self.blocks):
            block: TopDown = block
            top_down_result = block(x, skips[-(i + 1)], first)
            x = top_down_result.x
            nested_z_list.append(top_down_result.z)
            nested_mu_list.append(top_down_result.mu)
            nested_std_list.append(top_down_result.std)

            kls += torch.flatten(top_down_result.kls)
            first = False
            x = self.unpool(x)

        top_down_result = self.endblock(x, skips[0], first)
        x = top_down_result.x
        nested_z_list.append(top_down_result.z)
        nested_mu_list.append(top_down_result.mu)
        nested_std_list.append(top_down_result.std)

        x = self.conv(x)
        self.kls = torch.flatten(kls)
        return ModelOutput(
            reconstruction=x,
            z=nested_z_list,
            mu=nested_mu_list,
            std=nested_std_list
        )


class VdVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        architecture = VdvaeArchitecture(
            arch_config=config.arch_config,
            encoder=VdVaeEncoder(config.arch_config),
            decoder=VdVaeDecoder(config.arch_config)
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )


class SeqVdVae(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        # noinspection PyTypeChecker
        architecture = SeqVdvaeArchitecture(
            arch_config=config.arch_config,
            encoder=VdVaeEncoder(config.arch_config),
            decoder=VdVaeDecoder(config.arch_config)
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )
