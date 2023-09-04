from enum import auto

import torch
from torch.nn import Module
from torchvision.transforms.v2 import Normalize

from unitorchplate.data.datamodules import DataModule
from unitorchplate.models.modelmodule import ModelConfig
from unitorchplate.utils.processing.basic_transformations import Transformations


class ChannelNormalize(Normalize):
    """
    Normalize a tensor image using the torchvision Normalize module with automatically inferred channel-wise values for mean and std.

    Also using imagenet values for mean and std is possible by setting use_defaults to True.
    """

    def __init__(self, data_module: DataModule, use_defaults: bool = False, inplace: bool = False):
        self.is_sequence = data_module.config.dataset_config.return_sequence
        self.channel_first = data_module.config.dataset_config.channel_first

        if not self.channel_first:
            raise NotImplementedError("ChannelNormalize currently only supports channel first datasets.")

        if use_defaults:
            channel_means = []
            channel_stds = []
            channel_first = data_module.config.dataset_config.channel_first
            if self.is_sequence:
                channel_position = 1 if channel_first else -1
            else:
                channel_position = 0 if channel_first else -1
            non_channel_dims = [d for d in range(len(data_module.train_dataset[0].shape)) if d != channel_position]
            for image in data_module.train_dataset:
                channel_means.append(image.mean(dim=non_channel_dims))
                channel_stds.append(image.std(dim=non_channel_dims))
            mean = torch.stack(channel_means).mean(dim=0)
            std = torch.stack(channel_stds).mean(dim=0)
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        super().__init__(mean=mean, std=std, inplace=inplace)


class VisionTransformations(Transformations):
    """
    Vision transformations that can be applied to a tensor.
    """
    CH_NORMALIZE = auto()

    def instance(self, config: ModelConfig, data_module: DataModule) -> Module:
        match self:
            case VisionTransformations.CH_NORMALIZE:
                return ChannelNormalize(data_module)
