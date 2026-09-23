"""SYNTHETIC TEST FIXTURE ONLY.
A clean-room, native-interface-shaped Restormer-style model. No prior-reader
weights, dataset, original source, or original RNG trajectory are reproduced.
NEVER use it as the user's historical/native B0 or for reported patient training.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,c:int,heads:int=1):
        super().__init__()
        self.num_heads=heads
        self.temperature=nn.Parameter(torch.ones(heads,1,1))
        self.qkv=nn.Conv2d(c,3*c,1,bias=False)
        self.qkv_dwconv=nn.Conv2d(3*c,3*c,3,padding=1,groups=3*c,bias=False)
        self.project_out=nn.Conv2d(c,c,1,bias=False)
    def forward(self,x:Tensor)->Tensor:
        b,c,h,w=x.shape
        q,k,v=(t.reshape(b,self.num_heads,c//self.num_heads,h*w)
               for t in self.qkv_dwconv(self.qkv(x)).chunk(3,1))
        q,k=F.normalize(q,dim=-1),F.normalize(k,dim=-1)
        a=((q@k.transpose(-1,-2))*self.temperature).softmax(-1)
        return self.project_out((a@v).reshape(b,c,h,w))

class Norm(nn.Module):
    def __init__(self,c:int):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(1,c,1,1))
        self.bias=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x:Tensor)->Tensor:
        var,mean=torch.var_mean(x,dim=1,unbiased=False,keepdim=True)
        return (x-mean)*torch.rsqrt(var+1e-5)*self.weight+self.bias

class Block(nn.Module):
    def __init__(self,c:int,heads:int=1):
        super().__init__()
        hidden=int(c*2.66)
        self.norm1,self.norm2=Norm(c),Norm(c)
        self.attn=Attention(c,heads)
        self.project_in=nn.Conv2d(c,2*hidden,1,bias=False)
        self.dwconv=nn.Conv2d(2*hidden,2*hidden,3,padding=1,groups=2*hidden,bias=False)
        self.project_out=nn.Conv2d(hidden,c,1,bias=False)
    def forward(self,x:Tensor)->Tensor:
        x=x+self.attn(self.norm1(x))
        a,b=self.dwconv(self.project_in(self.norm2(x))).chunk(2,1)
        return x+self.project_out(F.gelu(a)*b)

class NoPriorReader(nn.Module):
    def forward(self,x:Tensor,prior=None)->Tensor:
        if prior is not None: raise ValueError('Fixture has no prior.')
        return x

class NativeContractFixture(nn.Module):
    def __init__(self,dim:int=12,blocks:tuple=(1,1,1,1),refinement_blocks:int=3):
        super().__init__()
        if dim%2: raise ValueError('Fixture dim must be even.')
        def stage(c,h,n): return nn.Sequential(*(Block(c,h) for _ in range(n)))
        def down(c): return nn.Sequential(nn.Conv2d(c,c//2,3,padding=1,bias=False),nn.PixelUnshuffle(2))
        def up(c): return nn.Sequential(nn.Conv2d(c,2*c,3,padding=1,bias=False),nn.PixelShuffle(2))
        self.patch_embed=nn.Conv2d(5,dim,3,padding=1,bias=False)
        self.encoder_level1=stage(dim,1,blocks[0])
        self.down1_2=down(dim)
        self.encoder_level2=stage(2*dim,2,blocks[1])
        self.down2_3=down(2*dim)
        self.encoder_level3=stage(4*dim,4,blocks[2])
        self.down3_4=down(4*dim)
        self.latent=stage(8*dim,8,blocks[3])
        self.up4_3=up(8*dim)
        self.reduce_chan_level3=nn.Conv2d(8*dim,4*dim,1,bias=False)
        self.decoder_level3=stage(4*dim,4,blocks[2])
        self.up3_2=up(4*dim)
        self.reduce_chan_level2=nn.Conv2d(4*dim,2*dim,1,bias=False)
        self.decoder_level2=stage(2*dim,2,blocks[1])
        self.up2_1=up(2*dim)
        self.decoder_level1=stage(2*dim,1,blocks[0])
        self.refinement=stage(2*dim,1,refinement_blocks)
        self.output=nn.Conv2d(2*dim,1,3,padding=1,bias=False)
        self.reader_latent,self.reader_decoder=NoPriorReader(),NoPriorReader()
    def forward(self,y:Tensor,prior=None)->Tensor:
        if prior is not None: raise ValueError('Fixture forbids prior.')
        e1=self.encoder_level1(self.patch_embed(y))
        e2=self.encoder_level2(self.down1_2(e1))
        e3=self.encoder_level3(self.down2_3(e2))
        z=self.reader_latent(self.latent(self.down3_4(e3)),prior)
        d3=self.decoder_level3(self.reduce_chan_level3(torch.cat((self.up4_3(z),e3),1)))
        d3=self.reader_decoder(d3,prior)
        d2=self.decoder_level2(self.reduce_chan_level2(torch.cat((self.up3_2(d3),e2),1)))
        d1=self.decoder_level1(torch.cat((self.up2_1(d2),e1),1))
        return self.output(self.refinement(d1))

def make_full_fixture(**kwargs)->NativeContractFixture:
    return NativeContractFixture(dim=48,blocks=(4,6,6,8),refinement_blocks=4)
