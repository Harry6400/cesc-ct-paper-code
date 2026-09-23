"""Exact linear operators and low-rank second-moment algebra.

M is a conditional UNCENTERED second moment, not necessarily a covariance.
All patches are valid: no reflected/artificial reference pixels in statistics.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F

def patches(x: Tensor) -> Tensor:
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 3:
        raise ValueError('Expected [N,1,H,W], H,W >= 3.')
    return F.unfold(x, kernel_size=3).transpose(1, 2) # N,L,9

def patch_adjoint(v: Tensor, size: tuple[int, int]) -> Tensor:
    if v.ndim != 3 or v.shape[-1] != 9:
        raise ValueError('Expected [N,L,9].')
    return F.fold(v.transpose(1, 2), size, kernel_size=3)

def pool(x: Tensor, factor: int) -> Tensor:
    return x if factor == 1 else F.avg_pool2d(x, factor, factor)

def pool_adjoint(x: Tensor, factor: int) -> Tensor:
    if factor == 1:
        return x
    return x.repeat_interleave(factor, -2).repeat_interleave(factor, -1) / factor**2

def low_rank_solve(d: Tensor, u: Tensor, v: Tensor) -> Tensor:
    # Woodbury: inverse of diag(d)+UU^T, only rank x rank Cholesky systems.
    if d.shape != v.shape or u.shape[:-1] != d.shape:
        raise ValueError('Incompatible D/U/vector dimensions.')
    inv = d.reciprocal()
    du = inv.unsqueeze(-1) * u
    small = u.transpose(-2, -1) @ du
    eye = torch.eye(u.shape[-1], device=u.device, dtype=u.dtype)
    chol = torch.linalg.cholesky(small + eye)
    dv = inv * v
    rhs = u.transpose(-2, -1) @ dv.unsqueeze(-1)
    sol = torch.cholesky_solve(rhs, chol)
    return dv - (du @ sol).squeeze(-1)

def low_rank_logdet(d: Tensor, u: Tensor) -> Tensor:
    du = u / d.unsqueeze(-1)
    eye = torch.eye(u.shape[-1], device=u.device, dtype=u.dtype)
    chol = torch.linalg.cholesky(eye + u.transpose(-2, -1) @ du)
    return d.log().sum(-1) + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)

@dataclass
class LocalMetric:
    d: Tensor  # N,L,9
    u: Tensor  # N,L,9,r
    image_size: tuple[int, int]
    mode: str = 'full'

    def __post_init__(self):
        if self.mode not in ('full', 'diagonal', 'identity'):
            raise ValueError('Unknown metric mode.')

    def solve(self, v: Tensor) -> Tensor:
        if self.mode == 'identity':
            return v
        if self.mode == 'diagonal':
            # Preserve each marginal second moment; remove only off-diagonal terms.
            return v / (self.d + self.u.square().sum(-1))
        return low_rank_solve(self.d, self.u, v)

    def apply(self, x: Tensor) -> Tensor:
        # Constant 1/9 preserves self-adjointness; per-pixel fold/count would not.
        return patch_adjoint(self.solve(patches(x)), self.image_size) / 9.0

    def quadratic(self, x: Tensor) -> Tensor:
        return 0.5 * (x * self.apply(x)).flatten(1).sum(-1)

    def upper_bound(self) -> Tensor:
        # ||P||^2 <= 9. Full/Diagonal use exactly the same conservative bound.
        return self.d.amin(dim=(1,2)).reciprocal().reshape(-1,1,1,1)

    def detach(self) -> 'LocalMetric':
        return LocalMetric(self.d.detach(), self.u.detach(), self.image_size, self.mode)
