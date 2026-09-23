"""Deterministic synthetic fixtures. These are NOT Mayo/clinical data or a Restormer baseline."""
import torch
from torch import Tensor
import torch.nn.functional as F
from cesc.bridge import Bridge,Batch

class SyntheticBridge(Bridge):
    def __init__(self,config:dict):
        self.cfg=config;self.bc=config.get('bridge_config',{})
        self.size=int(self.bc.get('size',16));self.micro=int(config.get('micro_batch',4))
        self.train_count=int(self.bc.get('train_count',8));self.val_count=int(self.bc.get('val_count',3))
    def contract(self):
        return {'source':'SYNTHETIC_ONLY','fixed_b0_sha256':'SYNTHETIC_NOT_A_MODEL',
                'manifest_sha256':'SYNTHETIC_GENERATOR_V1',
                's_ct':self.cfg['model']['s_ct'],'metric_version':'DEMO_ONLY_ssim11_valid_range4096',
                'patient_split':'SYNTHETIC_ONLY','b0_prediction_policy':'analytic fixture, not a learned model'}
    def _batch(self,indices,epoch,validation):
        centers=[];bases=[];targets=[];patients=[];slices=[]
        for index in indices:
            g=torch.Generator().manual_seed(999+index+(0 if validation else 1000+epoch*100))
            h=self.size
            yy,xx=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,h),indexing='ij')
            target=(1000+50*torch.exp(-3*(xx.square()+yy.square()))+15*torch.sin(4*xx))[None,None]
            noise=torch.randn((1,1,h,h),generator=g)*10
            center=target+noise
            # Fixed deterministic pseudo-B0 with a modest systematic error.
            base=target+0.25*noise+1.5*torch.cos(3*yy)[None,None]
            centers.append(center);bases.append(base);targets.append(target)
            patients.append(f'demo_patient_{index%3}');slices.append(f'{index:04d}')
        return Batch(torch.cat(centers),torch.cat(bases),torch.cat(targets),patients,slices).validate()
    def train_batches(self,epoch):
        for start in range(0,self.train_count,self.micro):
            yield self._batch(range(start,min(start+self.micro,self.train_count)),epoch,False)
    def validation_batches(self):
        for i in range(self.val_count):yield self._batch([i],0,True)
    def assert_frozen_b0(self):return {'source':'SYNTHETIC_ONLY','frozen':True}
    def metrics(self,pred_hu:Tensor,target_hu:Tensor):
        p,t=pred_hu,target_hu
        e=p-t;mse=e.square().mean()
        psnr=10*torch.log10(4096.0**2/mse.clamp_min(1e-12))
        size=min(11,p.shape[-2],p.shape[-1]);size-=1-size%2
        x=torch.arange(size,device=p.device,dtype=p.dtype)-(size-1)/2
        g=torch.exp(-x.square()/(2*1.5**2));g=g/g.sum();win=(g[:,None]*g[None,:])[None,None]
        mu1=F.conv2d(p,win);mu2=F.conv2d(t,win)
        v1=F.conv2d(p*p,win)-mu1*mu1;v2=F.conv2d(t*t,win)-mu2*mu2
        cov=F.conv2d(p*t,win)-mu1*mu2
        c1=(0.01*4096)**2;c2=(0.03*4096)**2
        ssim=(((2*mu1*mu2+c1)*(2*cov+c2))/((mu1.square()+mu2.square()+c1)*(v1+v2+c2))).mean()
        return {'psnr':float(psnr),'ssim':float(ssim),'mae':float(e.abs().mean()),'bias':float(e.mean())}
