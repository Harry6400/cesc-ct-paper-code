"""Local-method spatial cross-attention; not an upstream paper reproduction."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalPriorReader(nn.Module):
    def __init__(self, channels, prior_channels=32, projection=32, heads=4):
        super().__init__()
        if projection % heads:
            raise ValueError("projection must divide into heads")
        self.heads, self.width = heads, projection // heads
        self.query = nn.Conv2d(channels, projection, 1)
        self.key = nn.Conv2d(prior_channels, projection, 1)
        self.value = nn.Conv2d(prior_channels, projection, 1)
        self.output = nn.Conv2d(projection, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, prior):
        if prior is None:
            return features
        b, _, h, w = features.shape
        prior = F.interpolate(prior, size=(h, w), mode="bilinear", align_corners=False)
        q = self.query(features).reshape(b, self.heads, self.width, h*w)
        def neighborhoods(layer):
            return F.unfold(layer(prior), 3, padding=1).reshape(b, self.heads, self.width, 9, h*w)
        k, v = neighborhoods(self.key), neighborhoods(self.value)
        logits = (q.unsqueeze(3).float() * k.float()).sum(2) / self.width**0.5
        valid = F.unfold(torch.ones(1, 1, h, w, device=features.device), 3, padding=1).bool()
        weights = logits.masked_fill(~valid[:, None], float("-inf")).softmax(2)
        update = (weights.unsqueeze(2) * v.float()).sum(3).reshape(b, -1, h, w)
        return features + self.output(update.to(features.dtype))
