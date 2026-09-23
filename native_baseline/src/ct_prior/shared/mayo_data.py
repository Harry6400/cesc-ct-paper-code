from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import hu_to_unit


TRAIN_PATIENTS = ("L067", "L096", "L109", "L143", "L192", "L286")
VALIDATION_PATIENTS = ("L291", "L310", "L333")
LOCKED_TEST_PATIENT = "L506"
EXPECTED_VALIDATION_SLICES = 801


def _manifest_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {
        "rodiff_ct_mayo_manifest_1",
        "rodiff_ct_mayo_manifest_development_v1",
    }:
        raise ValueError("unsupported Mayo manifest schema")
    records = payload.get("records", [])
    patient_ids = [str(record.get("patient_id")) for record in records]
    expected_record_count = 9 if schema == "rodiff_ct_mayo_manifest_development_v1" else 10
    if len(records) != expected_record_count or len(set(patient_ids)) != expected_record_count:
        raise ValueError(f"Mayo manifest must contain exactly one record for each of {expected_record_count} patients")
    expected = {
        "train": set(TRAIN_PATIENTS),
        "val": set(VALIDATION_PATIENTS),
    }
    if schema == "rodiff_ct_mayo_manifest_1":
        expected["locked_test"] = {LOCKED_TEST_PATIENT}
    seen = {role: set() for role in expected}
    for record in records:
        role = str(record.get("role"))
        if role not in expected:
            raise ValueError(f"unexpected Mayo role: {role}")
        if record.get("dataset") != "mayo_3mm_b30":
            raise ValueError("Mayo campaign is frozen to 3 mm B30")
        if list(record.get("shape_qd", [])) != list(record.get("shape_fd", [])):
            raise ValueError(f"QD/FD shape mismatch for {record.get('patient_id')}")
        seen[role].add(str(record.get("patient_id")))
    if seen != expected:
        raise ValueError(f"Mayo patient split mismatch: {seen}")
    validation_slices = sum(int(record["shape_qd"][0]) for record in records if record["role"] == "val")
    if validation_slices != EXPECTED_VALIDATION_SLICES:
        raise ValueError(f"Mayo validation inventory must contain 801 slices, got {validation_slices}")
    return records


def load_mayo_records(path: str | Path, role: str) -> list[dict[str, Any]]:
    if role not in {"train", "validation"}:
        raise ValueError("Mayo baseline loaders expose train/validation only")
    manifest_role = "val" if role == "validation" else role
    records = [record for record in _manifest_records(path) if record["role"] == manifest_role]
    if any(record["patient_id"] == LOCKED_TEST_PATIENT for record in records):
        raise RuntimeError("locked Mayo patient entered a development loader")
    return records


class _AuditedMayoVolumes:
    def __init__(self, audit_path: str | Path) -> None:
        self.audit_path = Path(audit_path)
        self._cache: dict[str, np.ndarray] = {}

    def load(self, record: dict[str, Any], key: str, role: str) -> np.ndarray:
        if role not in {"train", "validation"} or record["role"] == "locked_test":
            raise RuntimeError("Mayo pixel access refused outside train/validation")
        path = str(record[key])
        if LOCKED_TEST_PATIENT in path:
            raise RuntimeError("locked Mayo patient pixel access refused")
        if path not in self._cache:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            event = json.dumps({
                "role": role,
                "patient_id": record["patient_id"],
                "volume_role": key,
                "path": path,
            }, sort_keys=True) + "\n"
            descriptor = os.open(self.audit_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(descriptor, event.encode("utf-8"))
            finally:
                os.close(descriptor)
            array = np.load(path, mmap_mode="r")
            expected = tuple(int(value) for value in record["shape_qd" if key == "qd_path" else "shape_fd"])
            if tuple(array.shape) != expected:
                raise ValueError(f"Mayo volume shape mismatch: {array.shape} != {expected}")
            self._cache[path] = array
        return self._cache[path]


class MayoTrainingDataset(Dataset):
    """Shared per-center crop ledger for all five methods and the EDCNN fallback."""

    def __init__(self, manifest: str | Path, method: str, audit_path: str | Path, *,
                 patch_size: int = 128, seed: int = 123456, stage: str = "main") -> None:
        if patch_size != 128:
            raise ValueError("Mayo master patch size is frozen at 128")
        if stage not in {"main", "unad_pretrain"}:
            raise ValueError("unsupported Mayo training stage")
        if stage == "unad_pretrain" and method != "unad":
            raise ValueError("UNAD pretraining ledger is exclusive to UNAD")
        self.records = load_mayo_records(manifest, "train")
        self.method = method
        self.stage = stage
        self.patch_size = patch_size
        self.seed = int(seed)
        self.epoch = 0
        self.volumes = _AuditedMayoVolumes(audit_path)
        radius = 3 if stage == "unad_pretrain" else 2
        self.samples = [
            (record_index, center)
            for record_index, record in enumerate(self.records)
            for center in range(radius, int(record["shape_qd"][0]) - radius)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _coordinate(self, record: dict[str, Any], center: int) -> tuple[int, int]:
        key = f"mayo:{self.seed}:{self.epoch}:{record['patient_id']}:{center}:128"
        stable_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(stable_seed)
        _, height, width = map(int, record["shape_qd"])
        return int(rng.integers(0, height - 127)), int(rng.integers(0, width - 127))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, center = self.samples[int(index)]
        record = self.records[record_index]
        qd = self.volumes.load(record, "qd_path", "train")
        y, x = self._coordinate(record, center)
        spatial = np.s_[y:y + 128, x:x + 128]
        if self.stage == "unad_pretrain":
            source = qd[center][spatial][None]
            neighbors = np.stack([qd[z][spatial] for z in range(center - 3, center + 4) if z != center])
            return {
                "source": torch.from_numpy(np.ascontiguousarray(hu_to_unit(source))),
                "pretrain_target": torch.from_numpy(np.ascontiguousarray(hu_to_unit(neighbors))),
                "patient_id": record["patient_id"], "slice_id": center, "y": y, "x": x,
            }
        fd = self.volumes.load(record, "fd_path", "train")
        source = qd[center - 1:center + 2, y:y + 128, x:x + 128] if self.method == "corediff" else qd[center][spatial][None]
        target = fd[center][spatial][None]
        return {
            "source": torch.from_numpy(np.ascontiguousarray(hu_to_unit(source))),
            "target": torch.from_numpy(np.ascontiguousarray(hu_to_unit(target))),
            "patient_id": record["patient_id"], "slice_id": center, "y": y, "x": x,
        }


class MayoValidationDataset(Dataset):
    """All 801 registered validation slices; the locked-test record is never selected."""

    def __init__(self, manifest: str | Path, method: str, audit_path: str | Path) -> None:
        self.records = load_mayo_records(manifest, "validation")
        self.method = method
        self.volumes = _AuditedMayoVolumes(audit_path)
        self.samples = [
            (record_index, slice_id)
            for record_index, record in enumerate(self.records)
            for slice_id in range(int(record["shape_qd"][0]))
        ]
        if len(self.samples) != EXPECTED_VALIDATION_SLICES:
            raise ValueError("Mayo validation ledger changed")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, slice_id = self.samples[int(index)]
        record = self.records[record_index]
        qd = self.volumes.load(record, "qd_path", "validation")
        fd = self.volumes.load(record, "fd_path", "validation")
        if self.method == "corediff":
            z = [max(0, slice_id - 1), slice_id, min(len(qd) - 1, slice_id + 1)]
            source = np.stack([qd[item] for item in z])
        else:
            source = qd[slice_id][None]
        return {
            "source": torch.from_numpy(np.ascontiguousarray(hu_to_unit(source))),
            "target": torch.from_numpy(np.ascontiguousarray(hu_to_unit(fd[slice_id][None]))),
            "patient_id": record["patient_id"], "series_id": record["patient_id"],
            "slice_id": slice_id,
        }
