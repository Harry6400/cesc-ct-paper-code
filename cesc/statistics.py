import math
import torch
from torch import nn, Tensor
from .config import ModelConfig
from .networks import ContextPyramid
from .ops import LocalMetric, patches, low_rank_logdet, low_rank_solve

class StatisticsNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg=cfg
        self.context=ContextPyramid(cfg.stats_width)
        self.head=nn.Conv2d(cfg.stats_width,9*(1+cfg.rank),1)
        nn.init.normal_(self.head.weight,std=0.002)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            p=(cfg.initial_second_moment-cfg.d_min)/(cfg.d_max-cfg.d_min)
            self.head.bias[:9].fill_(math.log(p/(1-p)))
            # U must NOT start identically zero: d(UU^T)/dU=0 at U=0.
            self.head.bias[9:].normal_(0.0,0.01)
    def forward(self,c: Tensor,mode: str='full') -> LocalMetric:
        raw=self.head(self.context(c)[0])[...,1:-1,1:-1]
        n,_,h,w=raw.shape
        raw=raw.permute(0,2,3,1).reshape(n,h*w,-1)
        d=self.cfg.d_min+(self.cfg.d_max-self.cfg.d_min)*raw[...,:9].sigmoid()
        u=raw[...,9:].reshape(n,h*w,9,self.cfg.rank)
        return LocalMetric(d,u,tuple(c.shape[-2:]),mode)

def statistics_loss(metric: LocalMetric, standardized_error: Tensor) -> tuple[Tensor,dict]:
    e=patches(standardized_error)
    q=(e*low_rank_solve(metric.d,metric.u,e)).sum(-1)
    ld=low_rank_logdet(metric.d,metric.u)
    # Gaussian working score, constant omitted, normalized per vector component.
    loss=(q+ld).mean()/18.0
    with torch.no_grad():
        info={'working_nll':float(loss), 'quadratic_per_component':float(q.mean()/9),
              'mean_marginal_second_moment':float((metric.d+metric.u.square().sum(-1)).mean())}
    return loss,info
