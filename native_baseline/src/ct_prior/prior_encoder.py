import torch
from torch import nn


class PriorEncoder(nn.Module):
    def __init__(self, input_channels, normalize=False):
        super().__init__()
        blocks = []
        for width in (32, 48, 64):
            blocks.extend((nn.Conv2d(input_channels, width, 3, stride=2, padding=1), nn.SiLU()))
            input_channels = width
        self.body = nn.Sequential(*blocks, nn.Conv2d(64, 32, 1))
        self.norm = nn.LayerNorm(32) if normalize else None

    def forward(self, x):
        p = self.body(x)
        if self.norm is not None:
            p = self.norm(p.permute(0, 2, 3, 1).float()).permute(0, 3, 1, 2).contiguous()
        return p


class PriorDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(64, 64, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1)
        ) for _ in range(4)])
        self.output = nn.Conv2d(64, 32, 3, padding=1)

    def forward(self, x, timestep, condition):
        freq = torch.exp(-torch.arange(32, device=x.device).float() * (9.210340371976184/32))
        phase = timestep.float()[:, None] * freq[None]
        emb = self.time(torch.cat((phase.cos(), phase.sin()), 1))[:, :, None, None]
        h = self.input(torch.cat((x, condition), 1)) + emb
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)
