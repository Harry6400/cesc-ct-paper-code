"""Native Mayo bridge bound to the completed E07-origin B0 checkpoint.

The bridge deliberately exposes only the registered train/validation patients.
L506 is present in the contract as a locked identity and is never placed in a
loader.  Training crops follow the native epoch-keyed Mayo ledger.  Validation
uses the native 128-patch/Hann-stitch B0 path rather than a whole-image shortcut.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from cesc.audit import file_sha256, module_sha256
from cesc.bridge import Batch, Bridge


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SRC = ROOT / "native_baseline" / "src"
if str(NATIVE_SRC) not in sys.path:
    sys.path.insert(0, str(NATIVE_SRC))

from ct_prior.data import MayoFive  # noqa: E402
from ct_prior.evaluate import region_metrics  # noqa: E402
from ct_prior.restoration_backbone import model_to_hu  # noqa: E402
from ct_prior.shared.metrics import patch_metrics  # noqa: E402
from ct_v29_prod.model import build_v29_objective  # noqa: E402
from ct_v29_prod.validation import infer_image_v29  # noqa: E402


TRAIN_PATIENTS = ["L067", "L096", "L109", "L143", "L192", "L286"]
VAL_PATIENTS = ["L291", "L310", "L333"]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class NativeE07B0Bridge(Bridge):
    def __init__(self, config: dict):
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.micro_batch = int(config["micro_batch"])
        self.manifest = Path(config["manifest_path"])
        self.checkpoint = Path(config["fixed_b0_checkpoint"])
        expected_deployment_sha = config.get("fixed_b0_deployment_sha256")
        if expected_deployment_sha and file_sha256(self.checkpoint) != expected_deployment_sha:
            raise RuntimeError("B0 deployment checkpoint SHA-256 mismatch")
        self.audit_dir = Path(config["pixel_audit_dir"])
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "rodiff_ct_v29_deployment_v1" or payload.get("variant") != "B0":
            raise RuntimeError("fixed checkpoint is not the E07-origin v29 B0 deployment")
        if payload.get("training_checkpoint_sha256") != config["fixed_b0_training_sha256"]:
            raise RuntimeError("B0 deployment is not bound to the selected E190 training checkpoint")
        objective = build_v29_objective(payload["config"])
        objective.system.load_state_dict(payload["system"], strict=True)
        self.b0 = objective.system.eval().requires_grad_(False)
        device_index = int(os.environ.get("LOCAL_RANK", config.get("local_rank", 0)))
        self.device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
        self.b0.to(self.device)
        self._b0_graph_cache = {} if self.device.type == "cuda" else None
        self._b0_state_sha = module_sha256(self.b0)
        expected_state_sha = config.get("fixed_b0_state_sha256")
        if expected_state_sha and expected_state_sha != self._b0_state_sha:
            raise RuntimeError("fixed B0 loaded-state hash mismatch")
        native_config = {
            "seed": int(config["seed"]),
            "dataset": "mayo",
            "context_mode": "real_five",
            "paths": {"manifest": str(self.manifest)},
        }
        self.train_dataset = MayoFive(
            native_config, "train", self.audit_dir / f"pixel_access.train.rank{self.rank}.jsonl"
        )
        self.validation_dataset = MayoFive(
            native_config, "validation", self.audit_dir / f"pixel_access.validation.rank{self.rank}.jsonl"
        )

    def contract(self) -> dict:
        return {
            "source": "NATIVE_E07_ORIGIN_B0",
            "fixed_b0_sha256": file_sha256(self.checkpoint),
            "fixed_b0_training_sha256": self.config["fixed_b0_training_sha256"],
            "b0_state_sha256": self._b0_state_sha,
            "manifest_sha256": file_sha256(self.manifest),
            "native_code_sha256": _tree_sha256(NATIVE_SRC),
            "patient_split": "train6_val3_lockedtest1",
            "patients": {"train": TRAIN_PATIENTS, "val": VAL_PATIENTS, "test": ["L506"]},
            "main_series": "3mm_B30",
            "fixed_b0_selected_epoch": 190,
            "fixed_b0_selected_on_validation_only": True,
            "s_ct": float(self.config["model"]["s_ct"]),
            "hu_center": float(self.config["model"]["hu_center"]),
            "hu_scale": float(self.config["model"]["hu_scale"]),
            "metric_version": "mayo_v11_local_gaussian_ssim11_sigma1p5_patient_equal",
            "native_reconstruction_loss_verified": True,
            "native_loss_definition": "mean(abs(e/s_ct))+0.1*mean((e/s_ct)^2)",
            "normalization_source": "ct_prior.restoration_backbone.hu_to_model/model_to_hu",
            "b0_eval_domain": "train_online_patch;validation_native_128_hann_stitch",
            "native_seed": 123456,
            "locked_test_access": False,
        }

    def _raw_batch(self, rows: list[dict]) -> Batch:
        y = torch.stack([row["y"] for row in rows])
        target = torch.stack([row["target_hu"] for row in rows]).float()
        center = model_to_hu(y[:, 2:3]).detach().float()
        with torch.no_grad():
            if tuple(y.shape[-2:]) == (512, 512):
                if len(rows) != 1:
                    raise RuntimeError("native validation bridge requires one full slice per batch")
                b0, _ = infer_image_v29(
                    self.b0, y[0], patch_batch=1, graph_cache=self._b0_graph_cache
                )
                b0 = b0.unsqueeze(0)
            else:
                b0 = self.b0(y.to(self.device)).detach().cpu()
        return Batch(
            center.cpu(), b0.float().cpu(), target.cpu(),
            [str(row["patient_id"]) for row in rows],
            [f"{row['slice_id']}:{row['top']}:{row['left']}" for row in rows],
        )

    def train_batches(self, epoch: int):
        data_epoch = int(epoch) - 1
        self.train_dataset.set_epoch(data_epoch)
        usable = len(self.train_dataset) // 32 * 32
        order = np.random.default_rng(int(self.config["seed"]) + data_epoch).permutation(
            len(self.train_dataset)
        )[:usable]
        groups = [order[i:i + self.micro_batch] for i in range(0, usable, self.micro_batch)]
        for group_index, indices in enumerate(groups):
            if group_index % self.world_size != self.rank:
                continue
            yield self._raw_batch([self.train_dataset[int(index)] for index in indices])

    def validation_batches(self, indices=None):
        ordered = range(len(self.validation_dataset)) if indices is None else [int(index) for index in indices]
        if len(set(ordered)) != len(ordered):
            raise ValueError("validation indices must be unique")
        if any(index < 0 or index >= len(self.validation_dataset) for index in ordered):
            raise IndexError("validation index outside registered validation dataset")
        for index in ordered:
            row = self.validation_dataset[index]
            yield self._raw_batch([row])

    def metrics(self, pred_hu: torch.Tensor, target_hu: torch.Tensor) -> dict:
        values = patch_metrics((pred_hu + 1024.0) / 4096.0, (target_hu + 1024.0) / 4096.0)
        values.update(region_metrics(pred_hu, target_hu))
        values = {
            "psnr_db": values["psnr_hu"],
            "ssim": values["ssim_hu"],
            "mae_hu": values["mae_hu"],
            "signed_bias_hu": values["signed_bias_hu"],
            "high_density_mae_hu": values["high_density_mae_hu"],
            "high_density_bias_hu": values["high_density_bias_hu"],
            "high_gradient_mae_hu": values["high_gradient_mae_hu"],
        }
        return values

    def assert_frozen_b0(self):
        if self.b0.training or any(module.training for module in self.b0.modules()):
            raise RuntimeError("fixed B0 left eval mode")
        if any(parameter.requires_grad for parameter in self.b0.parameters()):
            raise RuntimeError("fixed B0 has trainable parameters")
        return True

    def verify_b0_hash(self):
        self.assert_frozen_b0()
        if module_sha256(self.b0) != self._b0_state_sha:
            raise RuntimeError("fixed B0 weights or buffers changed")
        return True
