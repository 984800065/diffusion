import torch
import loguru
from torch import nn
from torch.nn import functional as F
from typing import Optional

class DenoisingDiffusionProbabilisticModel(nn.Module):
    def __init__(
        self, 
        eps_model: nn.Module, 
        n_steps: int
    ):
        super().__init__()
        self.n_steps = n_steps
        self.eps_model = eps_model

        beta = torch.linspace(0.0001, 0.02, n_steps)
        alpha = (1 - beta)
        alpha_bar = torch.cumprod(alpha, dim=0)
        sigma2 = beta

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sigma2", sigma2)

    @staticmethod
    def gather(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x[t.long()]
        x = x.reshape(-1, 1, 1, 1)
        return x

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean = self.gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - self.gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps_theta = self.eps_model(xt, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)

        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gather(self.sigma2, t)
        if noise is None:
            noise = torch.randn_like(xt, device=xt.device)
        return mean + (var ** 0.5) * noise

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size= x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size, ), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(eps_theta, noise)