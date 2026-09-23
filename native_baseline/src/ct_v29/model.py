"""One native D3 insertion, shared evidence interface, separately ablatable A/B.
This adapter is NOT a production training runner. Paper method name has no version.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from typing import Iterator
import random
import numpy as np
import torch
from torch import Tensor,nn
from .common import isolated_cpu_rng,tensor_digest,validate_alpha,norm_summary
from .evidence import EvidenceCarrier,OrderedAxialTransport
from .ridge import EvidenceSpanRidge

VARIANTS={'B0','B05center','A','B','AB','A_UNORDERED','A_FORWARD','A_NO_REJECT','A_CENTER_VALUES',
          'B_DIAGONAL','AB_DIAGONAL','AB_UNORDERED','AB_CENTER_VALUES'}

@dataclass(frozen=True)
class Architecture:
    width:int=24
    b_hidden:int=16
    shared_seed:int=129000
    a_seed:int=129001
    b_seed:int=129002
    solver_chunk_size:int=4096
    schema:str='ct_v29_r2_20260921'

class V29Model(nn.Module):
    def __init__(self,native:nn.Module,variant:str,settings:Architecture|None=None,strict_native:bool=True):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(f'Unknown variant: {variant}')
        if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in native.parameters()):
            raise ValueError('Construct native and extensions on CPU FP32; move together afterwards')
        self.settings=settings or Architecture()
        if self.settings.schema!='ct_v29_r2_20260921':raise ValueError('Architecture identity mismatch')
        self.variant,self.backbone=variant,native
        required=['patch_embed','encoder_level1','down1_2','encoder_level2','down2_3','encoder_level3',
            'down3_4','latent','reader_latent','up4_3','reduce_chan_level3','decoder_level3','reader_decoder',
            'up3_2','reduce_chan_level2','decoder_level2','up2_1','decoder_level1','refinement','output']
        for key in required:
            if not hasattr(native,key):raise TypeError('Missing native contract member: '+key)
        if not isinstance(native.reduce_chan_level3,nn.Conv2d):raise TypeError('Inspect native D3 reducer')
        channels=native.reduce_chan_level3.out_channels
        if strict_native:
            depths={'encoder_level1':4,'encoder_level2':6,'encoder_level3':6,'latent':8,
                'decoder_level3':6,'decoder_level2':6,'decoder_level1':4,'refinement':4}
            if channels!=192 or any(len(getattr(native,k))!=v for k,v in depths.items()):
                raise ValueError('Frozen native contract D3=192; depth 4/6/6/8,6/6/4,4')
            if self.settings.width!=24 or self.settings.b_hidden!=16:
                raise ValueError('Frozen extension widths are 24/16; changes need a new experiment identity')
        # Reject attention wrappers with extra learned children, even though we do not modify attention.
        attn=native.refinement[2].attn
        if set(dict(attn.named_children()))!={'qkv','qkv_dwconv','project_out'}:
            raise TypeError('Factory is not pure native attention; do not stack onto existing candidates')
        self.native_initial_sha256=tensor_digest(native.state_dict())
        self.native_state_keys=tuple(native.state_dict())
        self.carrier=None;self.a29=None;self.b29=None
        native_only = variant in {'B0', 'B05center'}
        if not native_only:
            with isolated_cpu_rng(self.settings.shared_seed):
                self.carrier=EvidenceCarrier(channels,self.settings.width)
        has_a=variant.startswith('A')
        has_b=(variant.startswith('B') and not native_only) or variant.startswith('AB')
        if has_a:
            mode=('unordered' if variant.endswith('UNORDERED') else
                  'forward' if variant.endswith('FORWARD') else
                  'no_reject' if variant.endswith('NO_REJECT') else
                  'center_values' if variant.endswith('CENTER_VALUES') else 'ordered')
            with isolated_cpu_rng(self.settings.a_seed):
                self.a29=OrderedAxialTransport(self.settings.width,mode)
        if has_b:
            mode='diagonal' if variant.endswith('DIAGONAL') else 'full'
            with isolated_cpu_rng(self.settings.b_seed):
                self.b29=EvidenceSpanRidge(channels,self.settings.width,self.settings.b_hidden,mode,self.settings.solver_chunk_size)
        self.alpha_a=self.alpha_b=self.alpha_extension=1.
        self.collect_diagnostics=False;self.last_diagnostics={}
        if self.native_initial_sha256!=tensor_digest(self.native_only_state()):
            raise RuntimeError('Extension initialization altered native parameters')

    def native_only_state(self)->dict[str,Tensor]:
        state=self.backbone.state_dict()
        return {k:state[k] for k in self.native_state_keys}

    def correction(self,y:Tensor,d3:Tensor)->Tensor:
        aa,bb,gg=map(validate_alpha,(self.alpha_a,self.alpha_b,self.alpha_extension))
        if self.training and (aa,bb,gg)!=(1.,1.,1.):
            raise RuntimeError('Training alpha is always 1; variation is evaluation-only')
        if self.carrier is None or (gg==0 and not self.collect_diagnostics):return torch.zeros_like(d3)
        z,v=self.carrier.encode(y)
        raw=self.carrier.raw_evidence(v)
        evidence=raw
        if self.a29 is not None:
            transported=self.a29(z,v)
            evidence=transported if aa==1 else raw if aa==0 else raw+aa*(transported-raw)
        average=evidence.mean(1)
        fused=average
        if self.b29 is not None:
            constrained=self.b29(evidence,d3,v[:,2])
            fused=constrained if bb==1 else average if bb==0 else average+bb*(constrained-average)
        delta=self.carrier.output(fused)
        if delta.shape!=d3.shape:raise RuntimeError('D3 correction shape mismatch')
        if self.collect_diagnostics:
            self.last_diagnostics={'variant':self.variant,'alpha_a':aa,'alpha_b':bb,'alpha_extension':gg,
                'raw_evidence':norm_summary(raw),'selected_evidence':norm_summary(evidence),
                'latent_fused':norm_summary(fused),'native_correction':norm_summary(delta),
                'relative_correction_rms':(delta.detach().square().mean().sqrt()/d3.detach().square().mean().sqrt().clamp_min(1e-12)).item(),
                'a_off_means_raw_evidence':True,'b_off_means_mean_fusion':True}
        return gg*delta

    def forward(self,y:Tensor,prior=None)->Tensor:
        if prior is not None:raise ValueError('prior_mode=none; prior input forbidden')
        if y.ndim!=4 or y.shape[1]!=5 or any(s%8 for s in y.shape[-2:]):
            raise ValueError('Expected B,5,H,W with H,W divisible by 8')
        if self.carrier is None:return self.backbone(y,prior=None)
        n=self.backbone
        e1=n.encoder_level1(n.patch_embed(y))
        e2=n.encoder_level2(n.down1_2(e1))
        e3=n.encoder_level3(n.down2_3(e2))
        z=n.reader_latent(n.latent(n.down3_4(e3)),None)
        d3=n.decoder_level3(n.reduce_chan_level3(torch.cat((n.up4_3(z),e3),1)))
        d3=n.reader_decoder(d3,None)
        d3=d3+self.correction(y,d3)
        d2=n.decoder_level2(n.reduce_chan_level2(torch.cat((n.up3_2(d3),e2),1)))
        d1=n.decoder_level1(torch.cat((n.up2_1(d2),e1),1))
        return n.output(n.refinement(d1))

    def reconstruct(self,y:Tensor,prior=None,scale_hu:float=22.299220188880145)->Tensor:
        import math
        if not math.isfinite(float(scale_hu)) or scale_hu<=0:raise ValueError('Positive finite residual scale required')
        return y[:,2:3].float()*2048.+1024.-float(scale_hu)*self(y,prior).float()

    def metadata(self)->dict:
        return {'variant':self.variant,'architecture':asdict(self.settings),
            'native_initial_sha256':self.native_initial_sha256,'paper_method':'Ordered Evidence-Constrained Restoration',
            'reference_role':'historical_B0','paired_b0_gate':'USER_WAIVED','matched_fresh_b0_available':False,
            'native_attention_unchanged':True}


@contextmanager
def probe_mode(model:V29Model,alpha_a:float=1.,alpha_b:float=1.,*,alpha_extension:float=1.,diagnostics:bool=True,preserve_rng:bool=True)->Iterator[None]:
    aa,bb,gg=map(validate_alpha,(alpha_a,alpha_b,alpha_extension))
    modes={m:m.training for m in model.modules()}
    saved=(model.alpha_a,model.alpha_b,model.alpha_extension,model.collect_diagnostics)
    sub=[(m,m.collect_diagnostics) for m in (model.a29,model.b29) if m is not None]
    py,npstate=(random.getstate(),np.random.get_state()) if preserve_rng else (None,None)
    cpu=torch.get_rng_state().clone() if preserve_rng else None
    cuda=(torch.cuda.get_rng_state_all()
          if preserve_rng and torch.cuda.is_initialized() else None)
    try:
        model.eval();model.alpha_a,model.alpha_b,model.alpha_extension=aa,bb,gg
        model.collect_diagnostics=diagnostics
        for m,_ in sub:m.collect_diagnostics=diagnostics
        with torch.no_grad():yield
    finally:
        if preserve_rng:
            random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        model.alpha_a,model.alpha_b,model.alpha_extension,model.collect_diagnostics=saved
        for m,d in sub:m.collect_diagnostics=d
        for m,t in modes.items():m.training=t
