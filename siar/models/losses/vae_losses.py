from torch import Module
from torch.nn import MSELoss, KLDivLoss


class VaeLoss(Module):
    def __init__(self, beta=1.0, reduction='sum'):
        super(VaeLoss, self).__init__()
        self.beta = beta
        self.reconstruction_loss_fn = MSELoss(reduction=reduction)
        self.kl_loss_fn = KLDivLoss(beta, reduction=reduction)

    def forward(self, input_data, reconstructed, mu, log_var):
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed, input_data)
        kl_loss = self.kl_loss_fn(mu, log_var)
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss
