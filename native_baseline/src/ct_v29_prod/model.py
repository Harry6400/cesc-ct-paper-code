"""Production bridge from the frozen native CT runner to the v29-r2 operator."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ct_prior.system import CTSystem, image_loss
from ct_v29.common import tensor_digest
from ct_v29.model import Architecture, V29Model, VARIANTS, probe_mode

FORMAL_VARIANTS = {"B0", "A", "B", "AB", "B05center"}
NATIVE_ONLY_VARIANTS = {"B0", "B05center"}
BASE_SEED = 123456
SCALE_HU = 22.299220188880145


def is_extension_key(key: str) -> bool:
    dotted = "." + key
    return any(token in dotted for token in (".carrier.", ".a29.", ".b29."))


def state_sha256(state: Mapping[str, torch.Tensor], *, native_only: bool = False) -> str:
    selected = {key: value for key, value in state.items()
                if not native_only or not is_extension_key(key)}
    return tensor_digest(selected)


class V29CTSystem(CTSystem):
    """Native HU reconstruction/loss path with evaluator-only v29 probes."""

    def forward(self, y: torch.Tensor, noise: torch.Tensor | None = None, *,
                alpha_a: float = 1.0, alpha_b: float = 1.0,
                alpha_extension: float = 1.0) -> torch.Tensor:
        if noise is not None or self.prior_mode != "none":
            raise ValueError("v29 accepts native five-slice input with prior_mode=none only")
        if (alpha_a, alpha_b, alpha_extension) == (1.0, 1.0, 1.0):
            return super().forward(y, None)
        if self.training or torch.is_grad_enabled():
            raise ValueError("non-unit alpha is evaluator-only and requires no_grad")
        with probe_mode(self.restorer, alpha_a=alpha_a, alpha_b=alpha_b,
                        alpha_extension=alpha_extension, diagnostics=False,
                        preserve_rng=not torch.cuda.is_current_stream_capturing()):
            return super().forward(y, None)


class V29Objective(nn.Module):
    def __init__(self, system: V29CTSystem) -> None:
        super().__init__()
        self.system = system
        self.teacher = None

    def forward(self, y: torch.Tensor, target_hu: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.system(y)
        image = image_loss(prediction, target_hu, self.system.scale_hu)
        return {"loss": image, "image": image, "prediction": prediction}


def build_v29_objective(config: Mapping) -> V29Objective:
    variant = str(config["variant"])
    if variant not in VARIANTS:
        raise ValueError(f"unknown v29 variant: {variant}")
    torch.manual_seed(BASE_SEED)
    raw = dict(config["backbone"])
    raw.setdefault("ffn_expansion_factor", 2.66)
    raw.setdefault("LayerNorm_type", "WithBias")
    system = V29CTSystem(float(config["residual_scale_hu"]), "none", raw)
    native = system.restorer
    native_hash = tensor_digest(native.state_dict())
    settings = Architecture(
        width=int(config.get("a_width", 24)),
        b_hidden=int(config.get("b_hidden", 16)),
        shared_seed=int(config.get("shared_seed", 129000)),
        a_seed=int(config.get("a_seed", 129001)),
        b_seed=int(config.get("b_seed", 129002)),
        solver_chunk_size=int(config.get("solver_chunk_size", 4096)),
    )
    system.restorer = V29Model(native, variant, settings=settings,
                               strict_native=bool(config.get("strict_native", True)))
    if system.restorer.native_initial_sha256 != native_hash:
        raise RuntimeError("v29 installation changed native state")
    system.v29_variant = variant
    system.v29_native_initial_sha256 = native_hash
    return V29Objective(system)


def create_v29_optimizer(objective: nn.Module, config: Mapping):
    optimizer = torch.optim.AdamW(
        objective.parameters(), lr=float(config["primary_lr"]),
        betas=tuple(config["optimizer_betas"]), eps=float(config["optimizer_eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    references = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    ids = set(references)
    extension = {name: parameter for name, parameter in objective.named_parameters()
                 if is_extension_key(name)}
    missing = [name for name, parameter in objective.named_parameters()
               if parameter.requires_grad and id(parameter) not in ids]
    duplicates = len(references) - len(ids)
    if config["variant"] not in NATIVE_ONLY_VARIANTS and not extension:
        raise RuntimeError("v29 extension parameters are absent")
    if missing or duplicates:
        raise RuntimeError("optimizer coverage failure")
    return optimizer, {
        "all_parameters_covered": not missing,
        "missing_parameters": missing,
        "duplicate_parameter_references": duplicates,
        "extension_parameter_count": sum(parameter.numel() for parameter in extension.values()),
        "extension_parameter_names": sorted(extension),
    }
