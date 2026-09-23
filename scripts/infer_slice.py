"""Infer one CT slice using a frozen B0 and a trained CESC deployment.

The precomputed-B0 mode also permits a synthetic smoke test without patient data.
Load checkpoints only from trusted sources: PyTorch checkpoint loading uses pickle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cesc.audit import file_sha256  # noqa: E402
from cesc.config import ModelConfig  # noqa: E402
from cesc.model import CESC  # noqa: E402


def _image(path: str) -> torch.Tensor:
    image = np.load(path, allow_pickle=False)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"expected a finite [H,W] HU image: {path}")
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0)


def _native_b0(qd_path: str, checkpoint_path: str, expected_training_sha: str,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    native_src = ROOT / "native_baseline" / "src"
    sys.path.insert(0, str(native_src))
    from ct_prior.restoration_backbone import hu_to_model, model_to_hu
    from ct_v29_prod.model import build_v29_objective
    from ct_v29_prod.validation import infer_image_v29

    qd = np.load(qd_path, allow_pickle=False)
    if qd.shape != (5, 512, 512) or not np.isfinite(qd).all():
        raise ValueError("--qd-five-hu must be a finite [5,512,512] HU array")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "rodiff_ct_v29_deployment_v1" or payload.get("variant") != "B0":
        raise ValueError("checkpoint is not the registered native B0 deployment")
    if payload.get("training_checkpoint_sha256") != expected_training_sha:
        raise ValueError("B0 training checkpoint identity mismatch")
    objective = build_v29_objective(payload["config"])
    objective.system.load_state_dict(payload["system"], strict=True)
    b0_model = objective.system.to(device).eval().requires_grad_(False)
    qd_model = hu_to_model(torch.from_numpy(np.asarray(qd, dtype=np.float32)))
    with torch.no_grad():
        b0_hu, _ = infer_image_v29(b0_model, qd_model)
    center_hu = model_to_hu(qd_model[2:3]).unsqueeze(0)
    return center_hu, b0_hu.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, help="trusted CESC best_deployment.pt")
    parser.add_argument("--out", required=True, help="output 2-D .npy file in HU")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--qd-five-hu", help="one five-slice [5,512,512] QD HU .npy")
    inputs.add_argument("--center-hu", help="precomputed 2-D center QD HU .npy")
    parser.add_argument("--b0-hu", help="precomputed 2-D frozen-B0 HU .npy")
    parser.add_argument("--b0-checkpoint", help="trusted registered B0 deployment checkpoint")
    args = parser.parse_args()
    if args.qd_five_hu and (not args.b0_checkpoint or args.b0_hu):
        parser.error("--qd-five-hu requires --b0-checkpoint and excludes --b0-hu")
    if args.center_hu and (not args.b0_hu or args.b0_checkpoint):
        parser.error("--center-hu requires --b0-hu and excludes --b0-checkpoint")

    device = torch.device(args.device)
    deployment = torch.load(args.deployment, map_location="cpu", weights_only=False)
    schema = deployment.get("schema")
    if schema != "rodiff_ct_v31_deployment_v1" and deployment.get("format") != "cesc-rebuilt-epoch-v1":
        raise ValueError("unsupported CESC checkpoint format")
    provenance = deployment.get("provenance", {})
    if provenance.get("variant") != "full" or provenance.get("stage") != "correction":
        raise ValueError("deployment must contain trained Full correction weights")
    if args.qd_five_hu:
        expected = provenance.get("bridge_contract", {})
        if file_sha256(args.b0_checkpoint) != expected.get("fixed_b0_sha256"):
            raise ValueError("B0 checkpoint differs from CESC training provenance")
        center, b0 = _native_b0(
            args.qd_five_hu, args.b0_checkpoint,
            expected.get("fixed_b0_training_sha256"), device,
        )
    else:
        center, b0 = _image(args.center_hu), _image(args.b0_hu)
    if center.shape != b0.shape:
        raise ValueError("center and B0 HU images must have equal shape")
    model = CESC(ModelConfig(**provenance["model_config"]), "full")
    model.load_state_dict(deployment["model"], strict=True)
    model.to(device).set_stage("eval")
    with torch.no_grad():
        prediction = model(center.to(device), b0.to(device))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prediction[0, 0].cpu().numpy())
    print(json.dumps({"output": str(output), "shape": list(prediction.shape[-2:]),
                      "input_mode": "native_b0" if args.qd_five_hu else "precomputed_b0"}))


if __name__ == "__main__":
    main()
