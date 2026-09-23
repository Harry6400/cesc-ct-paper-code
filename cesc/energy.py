"""Convex conditional potential with an explicit analytic gradient.

For fixed context, R(delta;c)=sum_s mean_channels sum_pixels
w_s * [sqrt((K_s P_s delta - a_s)^2+eps^2)-eps].
The conditioner and targets do not change across unrolled steps.
"""
from dataclasses import dataclass
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from .config import ModelConfig
from .networks import ContextPyramid
from .ops import pool,pool_adjoint,LocalMetric

@dataclass
class PotentialState:
    targets: list[Tensor]
    weights: list[Tensor]
    kernels: list[Tensor]

class ConditionalPotential(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        self.cfg=cfg
        self.context=ContextPyramid(cfg.energy_width)
        self.target_heads=nn.ModuleList([nn.Conv2d(k*cfg.energy_width,cfg.feature_channels,1) for k in (1,2,3)])
        self.weight_heads=nn.ModuleList([nn.Conv2d(k*cfg.energy_width,cfg.feature_channels,1) for k in (1,2,3)])
        self.raw_kernels=nn.ParameterList([nn.Parameter(torch.randn(cfg.feature_channels,1,3,3)*0.1) for _ in range(3)])
        for head in self.target_heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def prepare(self,c:Tensor)->PotentialState:
        features=self.context(c)
        targets=[head(x) for head,x in zip(self.target_heads,features)]
        weights=[self.cfg.weight_max*head(x).sigmoid() for head,x in zip(self.weight_heads,features)]
        kernels=[k/(k.abs().sum((1,2,3),keepdim=True)+1e-6) for k in self.raw_kernels]
        return PotentialState(targets,weights,kernels)
    def value(self,delta:Tensor,state:PotentialState)->Tensor:
        values=torch.zeros(delta.shape[0],device=delta.device,dtype=delta.dtype)
        for factor,a,w,k in zip((1,2,4),state.targets,state.weights,state.kernels):
            z=F.conv2d(pool(delta,factor),k,padding=1)-a
            phi=(z.square()+self.cfg.potential_eps**2).sqrt()-self.cfg.potential_eps
            values=values+(w*phi).flatten(1).sum(-1)/self.cfg.feature_channels
        return values
    def gradient(self,delta:Tensor,state:PotentialState)->Tensor:
        result=torch.zeros_like(delta)
        for factor,a,w,k in zip((1,2,4),state.targets,state.weights,state.kernels):
            z=F.conv2d(pool(delta,factor),k,padding=1)-a
            influence=w*z/(z.square()+self.cfg.potential_eps**2).sqrt()
            grad=F.conv_transpose2d(influence,k,padding=1)/self.cfg.feature_channels
            result=result+pool_adjoint(grad,factor)
        return result
    def lipschitz_bound(self,state:PotentialState)->Tensor:
        # Young's inequality, exact average-pooling operator norm, phi'' <= 1/eps.
        bound=state.targets[0].new_zeros(())
        for factor,k in zip((1,2,4),state.kernels):
            k_bound=k.abs().sum((1,2,3)).square().sum()
            bound=bound+self.cfg.weight_max*k_bound/(self.cfg.potential_eps*self.cfg.feature_channels*factor**2)
        return bound

class FiniteStepCorrection(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__();self.cfg=cfg
        self.potential=ConditionalPotential(cfg)
        self.step_logits=nn.Parameter(torch.zeros(cfg.steps))
    def energy(self,delta:Tensor,metric:LocalMetric,state:PotentialState)->Tensor:
        return metric.quadratic(delta)+0.5*self.cfg.anchor*delta.square().flatten(1).sum(-1)+self.potential.value(delta,state)
    def forward(self,c:Tensor,metric:LocalMetric,return_trace:bool=False):
        state=self.potential.prepare(c)
        delta=c[:,:1]*0
        bound=metric.upper_bound()+self.cfg.anchor+self.potential.lipschitz_bound(state)
        trace=[]
        if return_trace: trace.append(self.energy(delta,metric,state))
        for raw in self.step_logits:
            eta=1.8*raw.sigmoid()/bound
            grad=metric.apply(delta)+self.cfg.anchor*delta+self.potential.gradient(delta,state)
            delta=delta-eta*grad
            if return_trace: trace.append(self.energy(delta,metric,state))
        return (delta,trace) if return_trace else delta
