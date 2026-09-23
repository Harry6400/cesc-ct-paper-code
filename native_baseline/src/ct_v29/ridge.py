"""B: context-requested, evidence-span ridge correction.
The Gram matrix is latent evidence geometry, NEVER a CT noise covariance.
"""
from __future__ import annotations
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from .common import norm_summary


def ridge_projection(evidence:Tensor,request:Tensor,regularization:Tensor,*,
                     diagonal:bool=False,chunk_size:int=4096)->tuple[Tensor,Tensor]:
    """E: B,K,C,H,W, r: B,C,H,W, lambda: B,1,H,W.
    Solve (E^T E/C + lambda I) w = E^T r/C; return Ew and w.
    Exact math uses positive lambda, no explicit inverse, no detached solver.
    With full Gram, Ew lies in span(E) and ||Ew||_2 <= ||r||_2 pointwise.
    Diagonal is a CONTROL and does not have the latter guarantee.
    """
    if (evidence.ndim!=5 or request.ndim!=4 or regularization.ndim!=4
        or evidence.shape[0]!=request.shape[0] or evidence.shape[2:]!=request.shape[1:]
        or regularization.shape!=(request.shape[0],1,*request.shape[-2:]) or chunk_size<=0):
        raise ValueError('Bad evidence/request/regularization shape or chunk size')
    if evidence.dtype not in (torch.float32,torch.float64) or request.dtype!=evidence.dtype or regularization.dtype!=evidence.dtype:
        raise ValueError('Ridge requires consistent FP32 or FP64; formal training is FP32')
    # Never silently jitter/retry or fall back to diagonal on a failed solve.
    if not torch.isfinite(regularization).all() or (regularization<=0).any():
        raise ValueError('Regularization must be finite and strictly positive')
    b,k,c,h,w=evidence.shape
    e=evidence.permute(0,3,4,2,1).reshape(-1,c,k)
    r=request.permute(0,2,3,1).reshape(-1,c,1)
    lam=regularization.permute(0,2,3,1).reshape(-1,1,1)
    eye=torch.eye(k,dtype=evidence.dtype,device=evidence.device).unsqueeze(0)
    corrections=[];weights=[]
    for start in range(0,e.shape[0],chunk_size):
        ee,rr,ll=e[start:start+chunk_size],r[start:start+chunk_size],lam[start:start+chunk_size]
        gram=ee.transpose(1,2)@ee/c
        rhs=ee.transpose(1,2)@rr/c
        if diagonal:
            coeff=rhs/(gram.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)+ll)
        else:
            system=(gram+gram.transpose(1,2))*.5+ll*eye
            chol=torch.linalg.cholesky(system)
            coeff=torch.cholesky_solve(rhs,chol)
        corrections.append(ee@coeff);weights.append(coeff)
    delta=torch.cat(corrections,0).reshape(b,h,w,c).permute(0,3,1,2).contiguous()
    weight=torch.cat(weights,0).reshape(b,h,w,k).permute(0,3,1,2).contiguous()
    return delta,weight


class EvidenceSpanRidge(nn.Module):
    def __init__(self,native_channels:int=192,width:int=24,hidden:int=16,
                 mode:str='full',chunk_size:int=4096):
        super().__init__()
        if min(native_channels,width,hidden,chunk_size)<=0 or mode not in {'full','diagonal'}:
            raise ValueError('Invalid ridge configuration')
        self.width,self.native_channels,self.mode,self.chunk_size=width,native_channels,mode,chunk_size
        self.context=nn.Conv2d(native_channels,width,1,bias=False)
        self.request_net=nn.Sequential(nn.Conv2d(2*width,width,1,bias=False),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1,groups=width,bias=False),nn.GELU(),
            nn.Conv2d(width,width,1,bias=False))
        self.ridge_net=nn.Sequential(nn.Conv2d(2*width,hidden,1),nn.GELU(),nn.Conv2d(hidden,1,1))
        self.collect_diagnostics=False;self.last_diagnostics={}

    def forward(self,evidence:Tensor,native:Tensor,center:Tensor)->Tensor:
        if native.ndim!=4 or native.shape[1]!=self.native_channels or center.shape!=(native.shape[0],self.width,*native.shape[-2:]):
            raise ValueError('Bad native/center feature shape')
        context=self.context(native)
        conditioning=torch.cat((context,center),1)
        request=self.request_net(conditioning)
        # Scale does NOT depend on the evidence-use mass: rejection must not be
        # undone by proportionately shrinking lambda with the rejected columns.
        scale=center.square().mean(1,keepdim=True)+1e-4
        regularization=1e-6+scale*(.05+F.softplus(self.ridge_net(conditioning)))
        correction,weights=ridge_projection(evidence,request,regularization,
            diagonal=self.mode=='diagonal',chunk_size=self.chunk_size)
        if self.collect_diagnostics:
            en=evidence.detach().permute(0,3,4,2,1)
            gram=en.transpose(-2,-1)@en/self.width
            diagonal=gram.diagonal(dim1=-2,dim2=-1)
            offdiag=gram-torch.diag_embed(diagonal)
            dr=correction.detach().square().sum(1).sqrt()
            rr=request.detach().square().sum(1).sqrt()
            ratios=dr/rr.clamp_min(1e-12)
            self.last_diagnostics={'mode':self.mode,'request':norm_summary(request),'correction':norm_summary(correction),
                'lambda_min':regularization.detach().min().item(),'lambda_mean':regularization.detach().mean().item(),
                'lambda_max':regularization.detach().max().item(),'weight_abs_max':weights.detach().abs().max().item(),
                'request_retention_ratio_max':ratios.max().item(),
                'contraction_violation_max':(dr-rr).clamp_min(0).max().item(),
                'gram_offdiagonal_rms':offdiag.square().mean().sqrt().item(),
                'not_noise_covariance':True,'guarantee_scope':'latent_before_learned_output_projection'}
        return correction
