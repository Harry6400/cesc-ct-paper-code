from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .contract import DATASET_ID, EXPECTED_PATIENTS, EXPECTED_SERIES, read_json


HU_MIN = -1024.0
HU_MAX = 3072.0
HU_RANGE = HU_MAX - HU_MIN


def hu_to_unit(array: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(array, dtype=np.float32), HU_MIN, HU_MAX) - HU_MIN) / HU_RANGE


def unit_to_hu(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.float() * HU_RANGE + HU_MIN).clamp(HU_MIN, HU_MAX)


def load_pairs(volume_manifest: str | Path, split_manifest: str | Path, role: str) -> list[dict[str, Any]]:
    if role not in {"train", "validation"}:
        raise ValueError("LIDC baseline loader exposes only train/validation")
    split = read_json(split_manifest)
    if split.get("dataset_name") != DATASET_ID:
        raise ValueError("unexpected split dataset")
    if split.get("usage_policy") != "synthetic_robustness_only_by_default":
        raise ValueError("unexpected LIDC use policy")
    roles = split.get("roles", {})
    if {name: len(values) for name, values in roles.items()} != EXPECTED_PATIENTS:
        raise ValueError("patient split is not 808/101/101")
    all_patients = [str(item) for values in roles.values() for item in values]
    if len(all_patients) != len(set(all_patients)):
        raise ValueError("patient roles overlap")
    if any("L506" in item for item in all_patients):
        raise RuntimeError("L506 is forbidden")
    selected = set(map(str, roles[role]))
    records = read_json(volume_manifest)
    if isinstance(records, dict):
        records = records.get("records", records.get("rows", []))
    records = [row for row in records if row.get("dataset") == DATASET_ID]
    patient_key = str(split.get("patient_key", "source_patient_id"))
    doses = sorted({str(row["dose"]) for row in records})
    full_doses = [dose for dose in doses if dose.startswith("full")]
    low_doses = [dose for dose in doses if dose.startswith("quarter")]
    if len(full_doses) != 1 or len(low_doses) != 1:
        raise ValueError(f"expected one low/full dose pair, got {doses}")
    by_key = {(str(row["dose"]), str(row["patient_id"])): row for row in records}
    series_ids = sorted({
        str(row["patient_id"])
        for row in records
        if str(row.get(patient_key, row["patient_id"])) in selected
    })
    pairs: list[dict[str, Any]] = []
    for series_id in series_ids:
        low = by_key[(low_doses[0], series_id)]
        full = by_key[(full_doses[0], series_id)]
        patient_id = str(low.get(patient_key, series_id))
        shape = tuple(map(int, low["shape"]))
        if shape != tuple(map(int, full["shape"])) or shape[0] < 7 or min(shape[1:]) < 128:
            raise ValueError(f"invalid pair geometry for {series_id}: {shape}")
        paths = (str(low["npy_path"]), str(full["npy_path"]))
        if any("L506" in path for path in paths):
            raise RuntimeError("L506 path is forbidden")
        pairs.append({
            "patient_id": patient_id,
            "series_id": series_id,
            "low_path": paths[0],
            "full_path": paths[1],
            "shape": shape,
        })
    if len(pairs) != EXPECTED_SERIES[role] or len({pair["patient_id"] for pair in pairs}) != EXPECTED_PATIENTS[role]:
        raise ValueError(f"unexpected {role} inventory")
    return pairs


class LidcBaselineDataset(Dataset):
    """One shared 128-pixel master ledger with method-specific published views."""

    def __init__(
        self,
        volume_manifest: str | Path,
        split_manifest: str | Path,
        role: str,
        method: str,
        audit_path: str | Path,
        *,
        samples_per_series: int = 16,
        seed: int = 123456,
        stage: str = "main",
    ) -> None:
        if samples_per_series != 16:
            raise ValueError("campaign fixes 16 master patches per series")
        if stage not in {"main", "unad_pretrain"}:
            raise ValueError("unknown training stage")
        if stage == "unad_pretrain" and method != "unad":
            raise ValueError("pretraining stage is exclusive to UNAD")
        self.pairs = load_pairs(volume_manifest, split_manifest, role)
        self.role = role
        self.method = method
        self.stage = stage
        self.samples_per_series = samples_per_series
        self.seed = int(seed)
        self.epoch = 0
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._logged: set[str] = set()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_series

    def _load(self, path: str, pair: dict[str, Any], dose: str) -> np.ndarray:
        if path not in self._cache:
            if path not in self._logged:
                with self.audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "role": self.role,
                        "patient_id": pair["patient_id"],
                        "series_id": pair["series_id"],
                        "dose": dose,
                        "path": path,
                    }) + "\n")
                self._logged.add(path)
            self._cache[path] = np.load(path, mmap_mode="r")
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(path)
        return self._cache[path]

    def _coordinate(self, pair: dict[str, Any], sample_index: int) -> tuple[int, int, int]:
        epoch = self.epoch if self.role == "train" else 0
        base_seed = self.seed if self.role == "train" else 8123456
        key = f"{base_seed}:{epoch}:{pair['series_id']}:{sample_index}:5:128"
        stable_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(stable_seed)
        depth, height, width = pair["shape"]
        z_start = int(rng.integers(0, depth - 4))
        if self.stage == "unad_pretrain":
            center = min(max(z_start + 2, 3), depth - 4)
            z_start = center - 2
        return (
            z_start,
            int(rng.integers(0, height - 127)),
            int(rng.integers(0, width - 127)),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_index, sample_index = divmod(int(index), self.samples_per_series)
        pair = self.pairs[pair_index]
        z, y, x = self._coordinate(pair, sample_index)
        low_volume = self._load(pair["low_path"], pair, "synthetic_low")
        full_volume = self._load(pair["full_path"], pair, "clean_target")
        center = z + 2
        spatial = np.s_[y:y + 128, x:x + 128]
        if self.stage == "unad_pretrain":
            source = low_volume[center][spatial][None]
            neighbors = np.stack([low_volume[i][spatial] for i in range(center - 3, center + 4) if i != center])
            return {
                "source": torch.from_numpy(np.ascontiguousarray(hu_to_unit(source))),
                "pretrain_target": torch.from_numpy(np.ascontiguousarray(hu_to_unit(neighbors))),
                "patient_id": pair["patient_id"],
                "series_id": pair["series_id"],
                "sample_index": sample_index,
                "z": z,
                "y": y,
                "x": x,
            }
        if self.method == "corediff":
            source = low_volume[center - 1:center + 2, y:y + 128, x:x + 128]
        else:
            source = low_volume[center][spatial][None]
        target = full_volume[center][spatial][None]
        return {
            "source": torch.from_numpy(np.ascontiguousarray(hu_to_unit(source))),
            "target": torch.from_numpy(np.ascontiguousarray(hu_to_unit(target))),
            "patient_id": pair["patient_id"],
            "series_id": pair["series_id"],
            "sample_index": sample_index,
            "z": z,
            "y": y,
            "x": x,
        }


class SeriesGroupedDistributedSampler(Sampler[int]):
    """Shared deterministic ledger order with low memmap seek overhead.

    Every epoch permutes series and the 16 registered samples inside each
    series, then removes only the unavoidable global-batch remainder across
    distinct series. Rank slicing is deterministic and disjoint.
    """

    def __init__(self, dataset: LidcBaselineDataset, *, rank: int, world_size: int,
                 global_batch: int, seed: int) -> None:
        if dataset.samples_per_series != 16:
            raise ValueError("grouped sampler requires the registered 16 samples per series")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank")
        total = len(dataset)
        usable = total - total % int(global_batch)
        if usable % world_size:
            raise ValueError("usable ledger must divide across ranks")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_batch = int(global_batch)
        self.seed = int(seed)
        self.epoch = 0
        self.usable = usable

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.usable // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        series = torch.randperm(len(self.dataset.pairs), generator=generator).tolist()
        rows: list[list[int]] = []
        for pair_index in series:
            within = torch.randperm(16, generator=generator).tolist()
            rows.append([pair_index * 16 + sample for sample in within])
        remove = len(self.dataset) - self.usable
        for row in rows[:remove]:
            row.pop()
        order = [index for row in rows for index in row]
        if len(order) != self.usable:
            raise RuntimeError("grouped ledger length mismatch")
        return iter(order[self.rank::self.world_size])
