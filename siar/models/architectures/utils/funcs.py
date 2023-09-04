import torch


def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return torch.sum(
        -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2),
        dim=(1, 2, 3))