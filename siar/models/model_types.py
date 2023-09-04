from __future__ import annotations
from enum import auto
from typing import TYPE_CHECKING

import torch
from strenum import StrEnum

if TYPE_CHECKING:
    from siar.models.modelmodule import ModelConfig
    from unitorchplate.models.modelmodule import ModelModule


class ModelTypes(StrEnum):
    CUSTOM = auto()
    SimpleResVAE = auto()
    Sequential_SimpleResVAE = auto()
    NF_VAE = auto()
    VDVAE = auto()
    Sequential_VDVAE = auto()
    ResVAE = auto()
    ADV_VAE = auto()
    Sequential_ADV_VAE = auto()
    VDVAE_CS = auto()

    def cls(self) -> type[ModelModule]:
        match self:
            case ModelTypes.SimpleResVAE:
                from siar.models.architectures.simple_res_vae import SimpleResVae
                return SimpleResVae
            case ModelTypes.Sequential_SimpleResVAE:
                from siar.models.architectures.simple_res_vae import SimpleSeqResVae
                return SimpleSeqResVae
            case ModelTypes.NF_VAE:
                from siar.models.architectures.nf_vae import NfVae
                return NfVae
            case ModelTypes.VDVAE:
                from siar.models.architectures.vdvae import VdVae
                return VdVae
            case ModelTypes.Sequential_VDVAE:
                from siar.models.architectures.vdvae import SeqVdVae
                return SeqVdVae
            case ModelTypes.ResVAE:
                from siar.models.architectures.res_vae import ResVae
                return ResVae
            case ModelTypes.ADV_VAE:
                from siar.models.architectures.adv_vae import AdvVae
                return AdvVae
            case ModelTypes.Sequential_ADV_VAE:
                from siar.models.architectures.adv_vae import SeqAdvVae
                return SeqAdvVae
            case ModelTypes.CUSTOM:
                raise NotImplementedError("CUSTOM model type is not implemented")

    def instance(self, config: ModelConfig, preprocessing: torch.nn.Module) -> ModelModule | None:
        match self:
            case ModelTypes.SimpleResVAE:
                from siar.models.architectures.simple_res_vae import SimpleResVae
                return SimpleResVae(config=config, preprocessing=preprocessing)
            case ModelTypes.Sequential_SimpleResVAE:
                from siar.models.architectures.simple_res_vae import SimpleSeqResVae
                return SimpleSeqResVae(config=config, preprocessing=preprocessing)
            case ModelTypes.NF_VAE:
                from siar.models.architectures.nf_vae import NfVae
                return NfVae(config=config, preprocessing=preprocessing)
            case ModelTypes.VDVAE:
                from siar.models.architectures.vdvae import VdVae
                return VdVae(config=config, preprocessing=preprocessing)
            case ModelTypes.Sequential_VDVAE:
                from siar.models.architectures.vdvae import SeqVdVae
                return SeqVdVae(config=config, preprocessing=preprocessing)
            case ModelTypes.ResVAE:
                from siar.models.architectures.res_vae import ResVae
                return ResVae(config=config, preprocessing=preprocessing)
            case ModelTypes.ADV_VAE:
                from siar.models.architectures.adv_vae import AdvVae
                return AdvVae(config=config, preprocessing=preprocessing)
            case ModelTypes.Sequential_ADV_VAE:
                from siar.models.architectures.adv_vae import SeqAdvVae
                return SeqAdvVae(config=config, preprocessing=preprocessing)
            case ModelTypes.CUSTOM:
                return None
