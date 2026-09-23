"""Ordered local axial evidence. These are latent differences, not clean anatomy,
registered HU observations, or calibrated measurement uncertainties.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .common import sample_offset, norm_summary

OFFSETS=tuple((dy,dx) for dy in (-1,0,1) for dx in (-1,0,1))


def reciprocal_allocation(scores: Tensor, offsets=OFFSETS) -> dict[str,Tensor]:
    """Local scores B,D,H,W. Reverse probabilities are normalized at neighbor q.
    Invalid edges must be -inf; a self edge must exist at every location.
    """
    if scores.ndim!=4 or scores.shape[1]!=len(offsets):
        raise ValueError('Expected B,D,H,W scores and matching offsets')
    lf=scores.log_softmax(1)
    reverse=torch.stack([sample_offset(scores[:,i],-dy,-dx,float('-inf'))
                         for i,(dy,dx) in enumerate(offsets)],1)
    lb=reverse.log_softmax(1)
    back_at_q=torch.stack([sample_offset(lb[:,i],dy,dx,float('-inf'))
                          for i,(dy,dx) in enumerate(offsets)],1)
    joint=lf+back_at_q
    return {'conditional':joint.softmax(1),'forward':lf.exp(),
            'reciprocal_mass':joint.logsumexp(1,keepdim=True).exp()}


def use_mass(scores: Tensor, null_score: Tensor) -> Tensor:
    """Compare mean exponential valid evidence to a learned reject score.
    This sigmoid is an end-to-end use mass, NOT calibrated matchability.
    """
    count=torch.isfinite(scores).sum(1,keepdim=True)
    safe=torch.where(count>0,scores,torch.zeros_like(scores))
    logmean=safe.logsumexp(1,keepdim=True)-count.clamp_min(1).to(scores.dtype).log()
    return torch.where(count>0,torch.sigmoid(logmean-null_score),torch.zeros_like(null_score))


class EvidenceCarrier(nn.Module):
    """Shared encoder/value/readout used by A/B/AB with identical initialization.
    The carrier is an implementation interface, not a separately trained variant.
    """
    def __init__(self,out_channels:int=192,width:int=24):
        super().__init__()
        if min(width,out_channels)<=0:raise ValueError('Positive dimensions required')
        self.width,self.out_channels=width,out_channels
        self.encoder=nn.Sequential(nn.Conv2d(1,width,3,stride=2,padding=1,bias=False),nn.GELU(),
            nn.Conv2d(width,width,3,stride=2,padding=1,groups=width,bias=False),nn.GELU(),
            nn.Conv2d(width,width,1,bias=False),nn.GELU())
        self.value=nn.Conv2d(width,width,1,bias=False)
        self.output=nn.Conv2d(width,out_channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def encode(self,y:Tensor)->tuple[Tensor,Tensor]:
        if y.ndim!=4 or y.shape[1]!=5 or any(s%4 for s in y.shape[-2:]):
            raise ValueError('Expected B,5,H,W divisible by 4')
        b,_,h,w=y.shape
        zz=self.encoder(y.reshape(b*5,1,h,w))
        z=zz.reshape(b,5,self.width,h//4,w//4)
        v=self.value(zz).reshape_as(z)
        return z,v

    @staticmethod
    def raw_evidence(values:Tensor)->Tensor:
        # Slice/cat stays wholly on the input device and is CUDA-Graph safe;
        # list indexing materializes a CPU index tensor during capture.
        neighbors=torch.cat((values[:,:2],values[:,3:]),dim=1)
        return neighbors-values[:,2:3]


class OrderedAxialTransport(nn.Module):
    """A: produce FOUR columns, never collapse them before B sees them.
    Geometry uses signed slice INDEX only, with no physical-mm claim.
    """
    MODES={'ordered','unordered','forward','no_reject','center_values'}
    def __init__(self,width:int=24,mode:str='ordered'):
        super().__init__()
        if width<=0 or mode not in self.MODES:raise ValueError('Invalid A configuration')
        self.width,self.mode=width,mode
        self.descriptor=nn.Conv2d(width,width,1,bias=False)
        self.geometry=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
        self.null=nn.Sequential(nn.Conv2d(3*width+2,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        roles=(0,0,0,0) if mode=='unordered' else (-2,-1,1,2)
        self.register_buffer('_coordinates',torch.tensor(
            [[k/2.,abs(k)/2.,dy,dx] for k in roles for dy,dx in OFFSETS],
            dtype=torch.float32),persistent=False)
        self.register_buffer('_roles',torch.tensor(
            [[k/2.,abs(k)/2.] for k in roles],dtype=torch.float32),persistent=False)
        self.collect_diagnostics=False
        self.last_diagnostics={}

    def forward(self,z:Tensor,v:Tensor)->Tensor:
        if z.ndim!=5 or z.shape!=v.shape or z.shape[1:3]!=(5,self.width):
            raise ValueError('Expected matching B,5,C,H,W features/values')
        b,_,c,h,w=z.shape
        d=F.normalize(self.descriptor(z.reshape(b*5,c,h,w)),dim=1,eps=1e-6).reshape_as(z)
        z0,v0,d0=z[:,2],v[:,2],d[:,2]
        geometry=self.geometry(self._coordinates).reshape(4,9)
        valid=F.unfold(torch.ones_like(z0[:1,:1]),3,padding=1).reshape(1,9,h,w)>0
        columns=[];records=[]
        for i,k in enumerate((0,1,3,4)):
            dk=F.unfold(d[:,k],3,padding=1).reshape(b,c,9,h,w)
            score=(4*(d0.unsqueeze(2)*dk).sum(1)+geometry[i].view(1,9,1,1)).masked_fill(~valid,float('-inf'))
            allocation=reciprocal_allocation(score)
            p=allocation['forward'] if self.mode=='forward' else allocation['conditional']
            zn=(F.unfold(z[:,k],3,padding=1).reshape(b,c,9,h,w)*p.unsqueeze(1)).sum(2)
            role=self._roles[i].view(1,2,1,1).expand(b,2,h,w)
            null=self.null(torch.cat((z0,zn,(zn-z0).abs(),role),1))
            mass=use_mass(score,null)
            if self.mode=='no_reject':mass=torch.ones_like(mass)+0*mass  # retain a zero gradient edge for DDP
            source=v0 if self.mode=='center_values' else v[:,k]
            matched=(F.unfold(source,3,padding=1).reshape(b,c,9,h,w)*p.unsqueeze(1)).sum(2)
            # The target coordinate is p, so subtract v0(p), not v0(q).
            e=mass*(matched-v0)
            columns.append(e)
            if self.collect_diagnostics:
                records.append({'role_index':(-2,-1,1,2)[i], 'mean_use_mass':mass.detach().mean().item(),
                    'mean_entropy':(-(p*p.clamp_min(1e-30).log()).sum(1)).detach().mean().item(),
                    'mean_reciprocal_mass':allocation['reciprocal_mass'].detach().mean().item(),
                    'evidence':norm_summary(e)})
        evidence=torch.stack(columns,1)
        if self.collect_diagnostics:
            self.last_diagnostics={'mode':self.mode,'neighbors':records,'evidence':norm_summary(evidence),
                'geometry':'signed_index','not_calibrated_confidence':True,
                'not_clean_anatomy_difference':True}
        return evidence
