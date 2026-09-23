import math
import torch
from torch import nn
from torch.nn import functional as F
from .restoration_backbone import RestorationBackbone
from .prior_encoder import PriorEncoder, PriorDenoiser
from .prior_diffusion import PriorDiffusion

MODES = {"none", "global", "deterministic", "spatial_diffusion"}


def image_loss(prediction_hu, target_hu, scale):
    if prediction_hu.shape != target_hu.shape:
        raise ValueError("center target shape mismatch; broadcasting forbidden")
    error = (prediction_hu.float()-target_hu.float())/scale
    return error.abs().mean() + 0.1*error.square().mean()


class CTSystem(nn.Module):
    """Deployable model: deliberately has no teacher attribute or FD argument."""
    def __init__(self, scale_hu, prior_mode="none", backbone=None):
        super().__init__()
        if prior_mode not in MODES or not math.isfinite(scale_hu) or scale_hu <= 0:
            raise ValueError("invalid mode or residual scale")
        self.scale_hu, self.prior_mode = float(scale_hu), prior_mode
        self.backbone_config = backbone or {}
        self.restorer = RestorationBackbone(**self.backbone_config)
        self.condition = PriorEncoder(5)
        self.denoiser = PriorDenoiser()
        self.diffusion = PriorDiffusion()

    def read_prior(self, p):
        if self.prior_mode == "global":
            p = p.mean((-2, -1), keepdim=True).expand_as(p)
        return p

    def predict_prior(self, y, noise=None):
        c = self.condition(y)
        if self.prior_mode == "deterministic":
            p = torch.zeros_like(c)
            p = self.denoiser(p, torch.zeros(y.shape[0], dtype=torch.long, device=y.device), c)
            return p
        if noise is None:
            noise = torch.randn_like(c)
        if noise.shape != c.shape:
            raise ValueError("prior noise shape mismatch")
        return self.diffusion.rollout(self.denoiser, c, noise)

    def forward(self, y, noise=None):
        p = None if self.prior_mode == "none" else self.read_prior(self.predict_prior(y, noise))
        return self.restorer.reconstruct(y, p, self.scale_hu)


class IndependentObjective(nn.Module):
    """Complete objective for one independently initialized B0/D/G/M task."""

    def __init__(self, system):
        super().__init__()
        self.system = system
        self.teacher = PriorEncoder(6, normalize=True) if system.prior_mode != "none" else None
        self.requires_grad_(True)

    def teacher_prior(self, y, target_hu):
        target = (target_hu.float().clamp(-1024, 3072)-1024)/2048
        if target.shape != y[:, 2:3].shape:
            raise ValueError("teacher target must be the single center slice")
        return self.teacher(torch.cat((y, target), 1))

    def forward(self, y, target_hu):
        s = self.system
        if s.prior_mode == "none":
            prediction = s(y)
            image = image_loss(prediction, target_hu, s.scale_hu)
            return {"loss": image, "image": image, "prediction": prediction}
        teacher = self.teacher_prior(y, target_hu)
        oracle_prediction = s.restorer.reconstruct(y, s.read_prior(teacher), s.scale_hu)
        oracle_image = image_loss(oracle_prediction, target_hu, s.scale_hu)
        c = s.condition(y)
        if s.prior_mode == "deterministic":
            p = s.predict_prior(y)
            denoising = teacher.new_zeros(())
        else:
            p = s.diffusion.rollout(s.denoiser, c, torch.randn_like(teacher))
            t = torch.randint(4, (y.shape[0],), device=y.device)
            noisy = s.diffusion.q_sample(teacher, t, torch.randn_like(teacher))
            denoising = F.mse_loss(s.denoiser(noisy, t, c).float(), teacher.detach().float())
        distill = F.l1_loss(p.float(), teacher.detach().float())
        prediction = s.restorer.reconstruct(y, s.read_prior(p), s.scale_hu)
        image = image_loss(prediction, target_hu, s.scale_hu)
        loss = image + oracle_image + 0.1*distill + 0.1*denoising
        return {"loss": loss, "image": image, "oracle_image": oracle_image, "prior": distill,
                "denoise": denoising, "prediction": prediction}
