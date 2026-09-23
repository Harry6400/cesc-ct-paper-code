from dataclasses import dataclass, asdict
import hashlib
import json

@dataclass(frozen=True)
class ModelConfig:
    # Exact s_ct must be loaded from the native project; no rounded default in production.
    s_ct: float = 22.299
    hu_center: float = 1024.0
    hu_scale: float = 2048.0
    stats_width: int = 24
    energy_width: int = 32
    plain_width: int = 40
    rank: int = 2
    patch: int = 3
    d_min: float = 0.02
    d_max: float = 4.0
    initial_second_moment: float = 0.30
    feature_channels: int = 8
    steps: int = 3
    anchor: float = 0.1
    potential_eps: float = 0.1
    weight_max: float = 1.0
    stats_seed: int = 131001
    energy_seed: int = 131002
    plain_seed: int = 131003

    def __post_init__(self):
        if self.patch != 3 or self.rank not in (1, 2, 3, 4):
            raise ValueError('This reference supports valid 3x3 patches and rank 1..4.')
        if not 0 < self.d_min < self.initial_second_moment < self.d_max:
            raise ValueError('Require 0 < d_min < initial_second_moment < d_max.')
        if min(self.s_ct, self.hu_scale, self.anchor, self.potential_eps,
               self.weight_max, self.stats_width, self.energy_width,
               self.plain_width, self.steps, self.feature_channels) <= 0:
            raise ValueError('Scales, widths, step count and convexity constants must be positive.')

def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()
