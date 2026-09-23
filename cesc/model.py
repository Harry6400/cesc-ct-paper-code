import torch
from torch import nn,Tensor
from .config import ModelConfig
from .statistics import StatisticsNet
from .energy import FiniteStepCorrection
from .networks import ContextPyramid

class PlainCorrection(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__()
        self.context=ContextPyramid(cfg.plain_width)
        self.head=nn.Conv2d(cfg.plain_width,1,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,c:Tensor)->Tensor:
        return self.head(self.context(c)[0])

def context_tensor(center_hu:Tensor,b0_hu:Tensor,cfg:ModelConfig)->Tensor:
    if center_hu.shape!=b0_hu.shape or center_hu.ndim!=4 or center_hu.shape[1]!=1:
        raise ValueError('center_hu and b0_hu must both be [N,1,H,W].')
    if center_hu.requires_grad or b0_hu.requires_grad:
        raise ValueError('Native bridge must detach frozen B0 outputs and input observations.')
    return torch.cat([(center_hu-cfg.hu_center)/cfg.hu_scale,
                      (b0_hu-cfg.hu_center)/cfg.hu_scale,
                      (center_hu-b0_hu)/cfg.s_ct],dim=1)

class CESC(nn.Module):
    """Reference postprocessor: native B0 is deliberately external to this module."""
    def __init__(self,cfg:ModelConfig=ModelConfig(),variant:str='full'):
        super().__init__();self.cfg=cfg;self.variant=variant;self.stage='unconfigured'
        if variant not in ('full','diagonal','plain','b0'):
            raise ValueError('variant must be full, diagonal, plain, or b0.')
        # Isolated CPU RNG: construction cannot perturb the native B0/sample RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.stats_seed)
            self.statistics=StatisticsNet(cfg) if variant in ('full','diagonal') else None
            torch.manual_seed(cfg.energy_seed)
            self.correction=FiniteStepCorrection(cfg) if variant in ('full','diagonal') else None
            if variant=='plain':
                torch.manual_seed(cfg.plain_seed);self.correction=PlainCorrection(cfg)
    def set_stage(self,stage:str):
        if stage not in ('statistics','correction','eval'):
            raise ValueError('Unknown stage.')
        if stage=='statistics' and self.statistics is None:
            raise ValueError('No statistics stage for this variant.')
        self.stage=stage
        for p in self.parameters():p.requires_grad_(False)
        if stage=='statistics':
            for p in self.statistics.parameters():p.requires_grad_(True)
        elif stage=='correction' and self.correction is not None:
            for p in self.correction.parameters():p.requires_grad_(True)
        self.train(stage!='eval')
        return self
    def train(self,mode:bool=True):
        super().train(mode)
        if self.statistics is not None and self.stage!='statistics':self.statistics.eval()
        return self
    def statistic_metric(self,center_hu:Tensor,b0_hu:Tensor):
        if self.statistics is None:raise ValueError('No statistics module.')
        return self.statistics(context_tensor(center_hu,b0_hu,self.cfg),'full')
    def forward(self,center_hu:Tensor,b0_hu:Tensor,alpha:float=1.0,return_trace:bool=False):
        if not 0<=alpha<=1:raise ValueError('Audit alpha must lie in [0,1].')
        if self.variant=='b0' or alpha==0:
            return (b0_hu,[]) if return_trace else b0_hu
        c=context_tensor(center_hu,b0_hu,self.cfg)
        if self.variant=='plain':
            delta=self.correction(c);trace=[]
        else:
            with torch.no_grad():metric=self.statistics(c,self.variant).detach()
            result=self.correction(c,metric,return_trace)
            delta,trace=result if return_trace else (result,[])
        pred=b0_hu+(alpha*self.cfg.s_ct)*delta
        return (pred,trace) if return_trace else pred
    def parameter_report(self)->dict:
        total=sum(p.numel() for p in self.parameters())
        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'variant':self.variant,'addon_parameters':total,'trainable_parameters':trainable,
                'statistics_parameters':0 if self.statistics is None else sum(p.numel() for p in self.statistics.parameters()),
                'correction_parameters':0 if self.correction is None else sum(p.numel() for p in self.correction.parameters()),
                'native_b0_parameters':'NOT_INCLUDED'}
