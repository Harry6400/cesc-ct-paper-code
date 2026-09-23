from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import unit_to_hu


HU_RANGE = 4096.0


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def _ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Local Gaussian-window SSIM retained from the frozen native metric code."""
    x, y = x.float(), y.float()
    window = _gaussian_window(11, 1.5, x.device)
    mu_x = F.conv2d(x, window, padding=5)
    mu_y = F.conv2d(y, window, padding=5)
    sigma_x = F.conv2d(x * x, window, padding=5) - mu_x.square()
    sigma_y = F.conv2d(y * y, window, padding=5) - mu_y.square()
    sigma_xy = F.conv2d(x * y, window, padding=5) - mu_x * mu_y
    c1, c2 = (0.01 * HU_RANGE) ** 2, (0.03 * HU_RANGE) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def patch_metrics(pred_unit: torch.Tensor, target_unit: torch.Tensor) -> dict[str, float]:
    pred, target = unit_to_hu(pred_unit).cpu(), unit_to_hu(target_unit).cpu()
    error = pred - target
    mse = max(float(error.double().square().mean()), (HU_RANGE * HU_RANGE) * 1e-16)
    gy = F.pad(target[..., 1:, :] - target[..., :-1, :], (0, 0, 0, 1))
    gx = F.pad(target[..., :, 1:] - target[..., :, :-1], (0, 1, 0, 0))
    grad = torch.sqrt(gx.square() + gy.square())
    high_gradient = grad >= 50.0
    high_density = target > 240.0
    def masked(mask: torch.Tensor) -> tuple[float, float]:
        return ((float(error[mask].abs().mean()), float(error[mask].mean()))
                if bool(mask.any()) else (math.nan, math.nan))
    hd_mae, hd_bias = masked(high_density)
    hg_mae, _ = masked(high_gradient)
    return {"psnr_hu": 10.0 * math.log10(HU_RANGE * HU_RANGE / mse),
            "ssim_hu": _ssim(pred, target), "mae_hu": float(error.abs().mean()),
            "signed_bias_hu": float(error.mean()), "high_density_mae_hu": hd_mae,
            "high_density_bias_hu": hd_bias, "high_gradient_mae_hu": hg_mae}


class HierarchicalMetrics:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, patient_id: str, series_id: str, prediction: torch.Tensor,
            target: torch.Tensor, source_center: torch.Tensor) -> None:
        row = {"patient_id": patient_id, "series_id": series_id,
               **patch_metrics(prediction, target)}
        row.update({f"input_{key}": value for key, value in patch_metrics(source_center, target).items()})
        self.rows.append(row)

    @staticmethod
    def _mean(rows: list[dict[str, Any]]) -> dict[str, float]:
        keys = [key for key in rows[0] if key not in {"patient_id", "series_id"}]
        return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}

    def summarize(self) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("no validation rows")
        by_series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            by_series[(row["patient_id"], row["series_id"])].append(row)
        series_rows = [{"patient_id": patient, "series_id": series, **self._mean(rows)}
                       for (patient, series), rows in by_series.items()]
        by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in series_rows:
            by_patient[row["patient_id"]].append(row)
        patient_rows = [{"patient_id": patient, **self._mean(rows)} for patient, rows in by_patient.items()]
        return {"aggregation": "patch_to_series_to_patient_equal_weight",
                "patch_count": len(self.rows), "series_count": len(series_rows),
                "patient_count": len(patient_rows), "metrics": self._mean(patient_rows),
                "patients": patient_rows}
