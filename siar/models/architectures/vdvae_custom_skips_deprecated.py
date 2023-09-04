import torch
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor
from torch.nn import Module

from siar.models.architectures.base_architectures.variational_autoencoder import VdvaeArchitecture
from siar.models.architectures.utils.components import ResBlockVDVAE, TripleRes
from siar.models.architectures.utils.funcs import draw_gaussian_diag_samples, gaussian_analytical_kl
from siar.models.architectures.vdvae import VdvaeConfig
from siar.models.modelmodule import ModelConfig, VAEModelModule


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
        self.block = TripleRes(in_channels, hid_channels, out_channels)

    def sample(self, x, skip):

        p = self.resBlockP(x)
        pm, pv, xpp = p[:, :self.out_channels, ...], p[:, self.out_channels:self.out_channels * 2, ...], p[:,
                                                                                                         self.out_channels * 2:,
                                                                                                         ...]
        x = x + xpp

        if skip == "No_bool":
            z = draw_gaussian_diag_samples(pm, pv)
            kl = torch.zeros((x.shape[0]), device=x.device)

        else:
            qm, qv = self.resBlockQ(torch.cat([x, skip], dim=1)).chunk(2, dim=1)
            z = draw_gaussian_diag_samples(qm, qv)
            kl = gaussian_analytical_kl(qm, pm, qv, pv)

        return z, x, kl

    def forward(self, x, skip, first: bool = False):
        kls = torch.zeros((x.shape[0]), device=x.device)
        n = 3
        if first:
            n = 4
        for _ in range(n):
            z, x, kl = self.sample(x, skip)
            kls += kl
            x = x + self.conv11(z)
            x = self.block(x)
        self.kls = kls
        return x


class VdVaeEncoder(BaseEncoder):
    def __init__(self, config: VdvaeConfig):
        BaseEncoder.__init__(self)

        self.num_input_channels = config.input_dim[0]
        num_input_channels_res = config.resBlock_dim_in
        num_mid_channels_res = config.resBlock_dim_mid
        num_blocks = config.num_blocks
        self.skips_bool = config.skips_bool

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(self.num_input_channels, num_input_channels_res, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([
            TripleRes(num_input_channels_res, num_mid_channels_res, num_input_channels_res) for _ in range(num_blocks)
        ])
        self.endblock = nn.Sequential(
            ResBlockVDVAE(num_input_channels_res, num_mid_channels_res, num_input_channels_res),
            TripleRes(num_input_channels_res, num_mid_channels_res, num_input_channels_res))

    def forward(self, x: Tensor):
        skips = []
        x = self.conv(x)
        for n, block in enumerate(self.blocks):
            x = block(x)

            if self.skips_bool[n] == True:
                skips.append(x)
            else:
                skips.append("No_bool")
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
        num_blocks = config.num_blocks

        self.kls: Tensor | None = None

        self.blocks = torch.nn.ModuleList([
            TopDown(num_input_channels_res, num_mid_channels_res, num_input_channels_res) for _ in range(num_blocks)
        ])
        self.startblock = TopDown(num_input_channels_res, num_mid_channels_res, num_input_channels_res)
        self.unpool = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = torch.nn.Conv2d(num_input_channels_res, self.num_input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, encoder_output: dict):
        skips: list[Tensor] = encoder_output['skips']
        kls = torch.zeros((skips[-1].shape[0]), device=skips[-1].device)
        x = skips[-1]
        x = self.startblock(x, x, first=True)

        for i, block in enumerate(self.blocks):
            x = self.unpool(x)
            x = block(x, skips[-(i + 2)], first=False)
            kls += torch.flatten(block.kls)

        x = self.conv(x)

        self.kls = torch.flatten(kls)
        return ModelOutput(reconstruction=x)


class VdVae_CS(VAEModelModule):

    def __init__(self, config: ModelConfig, preprocessing: Module | None = None):
        architecture = VdvaeArchitecture(
            model_config=config.ae_config,
            encoder=VdVaeEncoder(config.ae_config),
            decoder=VdVaeDecoder(config.ae_config),
            sampler=None
        )
        super().__init__(
            architecture=architecture,
            config=config,
            preprocessing=preprocessing
        )
