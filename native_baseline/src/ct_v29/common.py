"""Small, explicit utilities; no dataset or training-policy ownership."""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
from typing import Iterator
import json
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@contextmanager
def isolated_cpu_rng(seed: int) -> Iterator[None]:
    # Do not torch.manual_seed(): it may also seed CUDA generators.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield

def sample_offset(x: Tensor, dy: int, dx: int, fill: float = 0.0) -> Tensor:
    """Return out[..., y,x] = in[..., y+dy,x+dx], with no wraparound."""
    h, w = x.shape[-2:]
    if abs(dy) >= h or abs(dx) >= w:
        return torch.full_like(x, fill)
    py, px = abs(dy), abs(dx)
    z = F.pad(x, (px, px, py, py), value=fill)
    return z[..., py+dy:py+dy+h, px+dx:px+dx+w]

def tensor_digest(state: dict[str, Tensor]) -> str:
    h = sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def json_digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode()).hexdigest()

def validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Probe alpha must be finite and in [0,1].')
    return alpha

def norm_summary(x: Tensor) -> dict[str, float]:
    y = x.detach().float()
    return {'rms': y.square().mean().sqrt().item(),
            'abs_max': y.abs().max().item()}

def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
