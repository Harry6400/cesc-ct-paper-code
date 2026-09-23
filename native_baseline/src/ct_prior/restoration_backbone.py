"""Mechanical Restormer adaptation: standardized residual, never RGB addition."""
import torch
from .official_restormer import Restormer
from .local_prior_reader import LocalPriorReader


def hu_to_model(x):
    return (x.float().clamp(-1024, 3072) - 1024.0) / 2048.0


def model_to_hu(x):
    return x.float() * 2048.0 + 1024.0


class RestorationBackbone(Restormer):
    def __init__(self, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), **kwargs):
        super().__init__(inp_channels=5, out_channels=1, dim=dim,
                         num_blocks=list(num_blocks), heads=list(heads),
                         num_refinement_blocks=num_refinement_blocks, **kwargs)
        self.reader_latent = LocalPriorReader(dim*8)
        self.reader_decoder = LocalPriorReader(dim*4)

    def forward(self, y, prior=None):
        if y.ndim != 4 or y.shape[1] != 5 or any(v % 8 for v in y.shape[-2:]):
            raise ValueError("expected B,5,H,W with H and W divisible by 8")
        e1 = self.encoder_level1(self.patch_embed(y))
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        z = self.reader_latent(self.latent(self.down3_4(e3)), prior)
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat((self.up4_3(z), e3), 1)))
        d3 = self.reader_decoder(d3, prior)
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat((self.up3_2(d3), e2), 1)))
        d1 = self.decoder_level1(torch.cat((self.up2_1(d2), e1), 1))
        return self.output(self.refinement(d1))

    def reconstruct(self, y, prior, scale_hu):
        residual = self(y, prior).float()
        return model_to_hu(y[:, 2:3]) - float(scale_hu) * residual
