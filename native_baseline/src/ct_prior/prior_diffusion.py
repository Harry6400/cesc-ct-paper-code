"""DiffIR pinned four-step schedule and x0 posterior mean, spatial layout."""
import torch
from torch import nn


class PriorDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        beta = torch.linspace(0.1**0.5, 0.99**0.5, 4, dtype=torch.float64).square()
        alpha_bar = (1-beta).cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        for name, value in {
            "betas": beta, "alpha_bar": alpha_bar,
            "posterior_x0": beta*previous.sqrt()/(1-alpha_bar),
            "posterior_xt": (1-previous)*(1-beta).sqrt()/(1-alpha_bar),
        }.items():
            self.register_buffer(name, value.float())

    def q_sample(self, prior, timestep, noise):
        a = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        return a.sqrt()*prior.float() + (1-a).sqrt()*noise.float()

    def rollout(self, denoiser, condition, noise, return_trace=False):
        x = noise.float()
        trace = []
        for t in (3, 2, 1, 0):
            ts = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
            clean = denoiser(x, ts, condition).float()
            x = self.posterior_x0[t]*clean + self.posterior_xt[t]*x
            trace.append(x)
        return (x, trace) if return_trace else x
