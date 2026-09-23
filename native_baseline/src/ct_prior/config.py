import hashlib
import json
import math
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
SCHEMA = "ct_spatial_prior_v8_independent_four_row_epoch200"
VARIANT_MODES = {"B0": "none", "D": "deterministic", "G": "global", "M": "spatial_diffusion"}
MAYO_B0_M_PROFILE = "mayo_b0_m_formal_v1"


def sha256(path):
    path = Path(path)
    if "L506" in str(path):
        raise RuntimeError("locked patient file access forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    for key, value in config["paths"].items():
        config["paths"][key] = value.replace("@package", str(PACKAGE))
    if config.get("calibration"):
        config["calibration"]["path"] = config["calibration"]["path"].replace("@package", str(PACKAGE))
    return config


def validate_config(config):
    if config.get("schema") != SCHEMA or config.get("test_locked") is not True:
        raise ValueError("invalid independent-training configuration contract")
    forbidden = {"stage_steps", "max_updates", "epoch_stages", "parent_checkpoint"}
    present = forbidden.intersection(config)
    if present:
        raise ValueError(f"fixed-stage fields are forbidden: {sorted(present)}")
    if config.get("dataset") not in {"mayo", "lidc"}:
        raise ValueError("invalid dataset")
    variant = config.get("variant")
    if variant not in VARIANT_MODES or config.get("prior_mode") != VARIANT_MODES[variant]:
        raise ValueError("four-row ablation switch mismatch")
    if config.get("seed") != 123456:
        raise ValueError("all variants must use the common registered seed")
    profile = config.get("protocol_profile", "v8_default")
    if profile not in {"v8_default", MAYO_B0_M_PROFILE}:
        raise ValueError("unknown protocol profile")
    if profile == MAYO_B0_M_PROFILE and not (config.get("dataset") == "mayo" and variant in {"B0", "M"}):
        raise ValueError("Mayo B0/M profile is restricted to Mayo B0 and M")
    if config.get("patch_size") != 128:
        raise ValueError("patch geometry changed")
    if config.get("micro_batch") != 4 or config.get("accumulation") != 8:
        raise ValueError("all variants must share micro_batch=4, accumulation=8")
    backbone = {"dim": 48, "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4, "heads": [1, 2, 4, 8]}
    if config.get("backbone") != backbone:
        raise ValueError("frozen architecture changed")
    expected = {
        "max_epochs": 200,
        "checkpoint_selection": "validation_only_patient_equal_psnr_then_ssim_then_mae",
        "learning_rate": 1e-4,
        "reduced_learning_rate": 2e-5,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"common training contract changed: {key}")
    profile_expected = {
        "validation_interval_epochs": 10 if profile == MAYO_B0_M_PROFILE else 5,
        "early_stopping": ("formal_baseline_two_stage_v1_mayo_no_epoch20_gate"
                           if profile == MAYO_B0_M_PROFILE else "formal_baseline_two_stage_v1"),
    }
    for key, value in profile_expected.items():
        if config.get(key) != value:
            raise ValueError(f"protocol profile contract changed: {key}")
    gate_enabled = config.get("epoch20_convergence_gate", True)
    if profile == MAYO_B0_M_PROFILE and gate_enabled is not False:
        raise ValueError("Mayo B0/M profile must disable the generic epoch20 convergence gate")
    if profile == "v8_default" and "epoch20_convergence_gate" in config:
        raise ValueError("epoch20_convergence_gate is only configurable in the Mayo B0/M profile")
    if config.get("validation_patch_streams", 1) != 1:
        raise ValueError("all variants must share validation execution")
    scale = config.get("residual_scale_hu")
    if config["dataset"] == "mayo" and scale != 22.299220188880145:
        raise ValueError("Mayo scale changed")
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError("invalid scale")


def verify_data_contract(config):
    for key, expected in config["manifest_sha256"].items():
        if sha256(config["paths"][key]) != expected:
            raise ValueError(f"{key} manifest hash mismatch")
    if config["dataset"] == "lidc":
        calibration = config.get("calibration")
        if not calibration:
            raise ValueError("LIDC train-only residual calibration required before training")
        if sha256(calibration["path"]) != calibration["sha256"]:
            raise ValueError("calibration hash mismatch")
        audit = json.loads(Path(calibration["path"]).read_text())
        if audit["role"] != "train" or audit["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError("calibration provenance mismatch")
        if audit["residual_scale_hu"] != config["residual_scale_hu"]:
            raise ValueError("calibration scale mismatch")
