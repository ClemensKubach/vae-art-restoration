from abc import ABC, abstractmethod

import torch
from torch.nn import Module


def sample_gauss(mu, std):
    # Reparametrization trick
    # Sample N(0, I)
    eps = torch.randn_like(std)
    return mu + eps * std, eps


class Sampler(Module, ABC):
    """Abstract class for samplers.

    They can be used to sample a latent vector z from a distribution, e.g. a Gaussian distribution.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, mu, std) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GaussSampler(Sampler):
    def __init__(self):
        super().__init__()

    def forward(self, mu, std):
        return sample_gauss(mu, std)
