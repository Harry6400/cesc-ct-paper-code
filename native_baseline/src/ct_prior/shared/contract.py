from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LIDC_SCHEMA = "rodiff_ct_lidc_baseline_config_v1"
UNIFIED_SCHEMA = "rodiff_ct_dual_dataset_baseline_config_v2"
METHODS = ("redcnn", "ctformer", "corediff", "unad", "dugan", "edcnn")
PRIMARY_METHODS = ("redcnn", "ctformer", "corediff", "unad", "dugan")
EXPECTED_PATIENTS = {"train": 808, "validation": 101, "test": 101}
EXPECTED_SERIES = {"train": 815, "validation": 102, "test": 101}
LIDC_DATASET_ID = "lidc_idri_synthetic_5to10_replay_v2"
MAYO_DATASET_ID = "mayo_3mm_b30_rodiff_v10b_protocol"
DATASET_ID = LIDC_DATASET_ID  # Backward-compatible import for the LIDC loader.
SELECTION_POLICY = "validation_only_patient_equal_psnr_then_ssim_then_mae"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any], method: str, *, allow_smoke: bool) -> None:
    dataset_id = config.get("dataset_id")
    schema = config.get("schema")
    if schema not in {LIDC_SCHEMA, UNIFIED_SCHEMA}:
        raise ValueError("unsupported baseline config schema")
    if schema == LIDC_SCHEMA and dataset_id != LIDC_DATASET_ID:
        raise ValueError("the v1 schema is LIDC-only")
    if method not in METHODS or method != config.get("method"):
        raise ValueError("method/config mismatch")
    if dataset_id not in {LIDC_DATASET_ID, MAYO_DATASET_ID}:
        raise ValueError("unexpected baseline dataset id")
    if config.get("role") not in {"train", "validation"}:
        raise ValueError("only train and validation roles are exposed")
    if config.get("test_locked") is not True:
        raise ValueError("LIDC test must remain locked")
    if config.get("random_initialization") is not True:
        raise ValueError("external baselines must start from random initialization")
    if config.get("official_weights_policy") != "smoke_only_never_train_init":
        raise ValueError("official weights policy is not frozen")
    if dataset_id == LIDC_DATASET_ID and int(config.get("samples_per_series", -1)) != 16:
        raise ValueError("LIDC samples_per_series must be 16")
    if dataset_id == MAYO_DATASET_ID:
        if config.get("train_ledger") != "all_valid_centers_one_seeded_128_crop_per_epoch":
            raise ValueError("Mayo must use the shared all-center training ledger")
        if config.get("validation_ledger") != "all_801_slices_three_patients":
            raise ValueError("Mayo must validate on all 801 registered slices")
    if int(config.get("master_patch_size", -1)) != 128:
        raise ValueError("master patch must be 128")
    if int(config.get("effective_batch", -1)) != 32:
        raise ValueError("effective batch must be 32")
    if int(config.get("micro_batch", 0)) * int(config.get("gradient_accumulation", 0)) != 32:
        raise ValueError("micro batch times accumulation must equal effective batch 32")
    if int(config.get("max_epochs", -1)) != 200:
        raise ValueError("main training cap must be 200 epochs")
    if int(config.get("validation_interval_epochs", -1)) != 5:
        raise ValueError("validation interval must be five epochs")
    if int(config.get("seed", -1)) != 123456 or int(config.get("validation_seed", -1)) != 8123456:
        raise ValueError("campaign seeds changed")
    if config.get("checkpoint_selection") != SELECTION_POLICY:
        raise ValueError("checkpoint selection must be validation-only and patient-equal")
    if config.get("formal_launch_gate") != "method_gpu_smoke_pass_and_config_freeze":
        raise ValueError("formal launch must remain behind the GPU-smoke/config-freeze gate")
    if method == "redcnn" and config.get("precision") != "fp32":
        raise ValueError("RED-CNN must use fp32")
    if method == "ctformer" and config.get("precision") != "fp32":
        raise ValueError("CTformer must use fp32")
    if method == "corediff" and int(config.get("nfe", -1)) != 10:
        raise ValueError("CoreDiff NFE must be 10")
    if method == "unad" and int(config.get("pretrain_epochs", -1)) != 40:
        raise ValueError("UNAD pretraining must be 40 epochs")
    if method == "dugan" and config.get("fallback") != "edcnn":
        raise ValueError("DU-GAN has only the EDCNN fallback")
    if method == "dugan":
        if int(config.get("discriminator_updates", -1)) != 2:
            raise ValueError("DU-GAN must preserve two discriminator updates per generator update")
        if int(config.get("discriminator_effective_batch", -1)) != 32:
            raise ValueError("DU-GAN discriminator effective batch must be 32")
    if method == "edcnn":
        if config.get("fallback_for") != "dugan":
            raise ValueError("EDCNN may only replace DU-GAN")
        if config.get("feature_loss_policy") != "fixed_imagenet_resnet50_loss_only":
            raise ValueError("EDCNN compound-loss feature policy changed")
    if not allow_smoke and config.get("run_kind") != "formal":
        raise ValueError("formal runner refuses smoke config")
    if dataset_id == LIDC_DATASET_ID and "L506" in json.dumps(config, ensure_ascii=False):
        raise RuntimeError("L506 identifier is forbidden in LIDC baseline configs")


def validate_bound_files(config: dict[str, Any]) -> dict[str, str]:
    paths = config["paths"]
    actual = {"manifest": sha256(paths["manifest"])}
    if config["dataset_id"] == LIDC_DATASET_ID:
        actual["split"] = sha256(paths["split_manifest"])
    expected = config["sha256"]
    if actual["manifest"] != expected["manifest"]:
        raise RuntimeError("volume manifest SHA mismatch")
    if "split" in actual and actual["split"] != expected["split_manifest"]:
        raise RuntimeError("LIDC split manifest SHA mismatch")
    return actual
