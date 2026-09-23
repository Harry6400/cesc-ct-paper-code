import torch
from torch import nn, Tensor
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width,width,3,padding=1), nn.SiLU(),
                                  nn.Conv2d(width,width,3,padding=1))
    def forward(self, x: Tensor) -> Tensor:
        return x + 0.1*self.body(x)

class ContextPyramid(nn.Module):
    """Small U-shaped conditioning CNN. No BN/dropout/global image statistics."""
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.SiLU(),ResidualBlock(width))
        self.down1 = nn.Sequential(nn.Conv2d(width,2*width,3,padding=1),nn.SiLU(),ResidualBlock(2*width))
        self.down2 = nn.Sequential(nn.Conv2d(2*width,3*width,3,padding=1),nn.SiLU(),ResidualBlock(3*width))
        self.up1 = nn.Sequential(nn.Conv2d(5*width,2*width,3,padding=1),nn.SiLU(),ResidualBlock(2*width))
        self.up0 = nn.Sequential(nn.Conv2d(3*width,width,3,padding=1),nn.SiLU(),ResidualBlock(width))
    def forward(self, c: Tensor) -> list[Tensor]:
        if c.ndim != 4 or c.shape[1] != 3 or c.shape[-2]%4 or c.shape[-1]%4:
            raise ValueError('Context must be [N,3,H,W] with H,W divisible by 4.')
        z0=self.stem(c)
        z1=self.down1(F.avg_pool2d(z0,2))
        z2=self.down2(F.avg_pool2d(z1,2))
        q1=self.up1(torch.cat([z1,F.interpolate(z2,scale_factor=2,mode='nearest')],1))
        q0=self.up0(torch.cat([z0,F.interpolate(q1,scale_factor=2,mode='nearest')],1))
        return [q0,q1,z2]
