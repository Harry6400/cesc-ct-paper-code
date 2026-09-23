"""Build hash-bound safety references without reading patient pixels."""

from __future__ import annotations

import hashlib
from pathlib import Path


METRIC_MAP = {
    "input_psnr_hu": "psnr_db",
    "input_ssim_hu": "ssim",
    "input_mae_hu": "mae_hu",
    "input_high_density_mae_hu": "high_density_mae_hu",
    "input_high_gradient_mae_hu": "high_gradient_mae_hu",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lidc_input_reference(source: dict, manifest_sha256: dict, source_path: Path) -> dict:
    summary = source["summary"]
    patient_rows = source["patients"]
    if summary.get("patients") != 101 or summary.get("series") != 102:
        raise ValueError("expected the registered 101-patient/102-series validation set")
    if summary.get("patches") != 1632 or len(patient_rows) != 101:
        raise ValueError("expected the registered 1632-patch validation ledger")
    patient_ids = [row["patient_id"] for row in patient_rows]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("duplicate patient in source validation report")

    def metrics(row: dict) -> dict:
        missing = set(METRIC_MAP) - set(row)
        if missing:
            raise ValueError(f"source report is missing input metrics: {sorted(missing)}")
        return {target: float(row[source_key]) for source_key, target in METRIC_MAP.items()}

    return {
        "schema": "ct_prior_safety_reference_v1",
        "kind": "whole",
        "evaluation_role": "deployment_validation",
        "reference_role": "fixed_synthetic_low_input",
        "aggregation": "slice_or_patch_to_series_to_patient_equal",
        "samples": int(summary["patches"]),
        "metrics": metrics(summary),
        "patients": {row["patient_id"]: metrics(row) for row in patient_rows},
        "manifest_sha256": manifest_sha256,
        "regions": "target>=200HU; top10percent gradients inside target>=-900HU",
        "source_report": str(source_path),
        "source_report_sha256": sha256(source_path),
        "pixel_accessed_by_conversion": False,
    }
