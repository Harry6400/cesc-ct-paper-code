"""Semantics-preserving v29 validation with measurable patch batching."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np
import torch

from ct_prior.evaluate import region_metrics, starts, summarize
from ct_prior.shared.metrics import patch_metrics

from .model import V29CTSystem


class FixedShapeGraphRunner:
    """Capture exactly one batch-one forward; caller preserves patch ordering."""

    def __init__(self, model: V29CTSystem, example: torch.Tensor,
                 alpha_a: float, alpha_b: float, alpha_extension: float = 1.0) -> None:
        if example.device.type != "cuda" or example.shape[0] != 1:
            raise ValueError("CUDA Graph requires one fixed CUDA patch")
        self.static_input = torch.empty_like(example)
        self.static_input.copy_(example)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model(self.static_input, alpha_a=alpha_a, alpha_b=alpha_b,
                      alpha_extension=alpha_extension)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = model(self.static_input, alpha_a=alpha_a, alpha_b=alpha_b,
                                       alpha_extension=alpha_extension)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.static_input.copy_(value)
        self.graph.replay()
        return self.static_output


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def infer_image_v29(
    model: V29CTSystem,
    y: torch.Tensor,
    *,
    alpha_a: float = 1.0,
    alpha_b: float = 1.0,
    alpha_extension: float = 1.0,
    patch: int = 128,
    patch_batch: int = 1,
    pin_memory: bool = False,
    graph_cache: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if patch_batch not in {1, 2, 4, 8, 16, 25}:
        raise ValueError("patch_batch must be one of 1/2/4/8/16/25")
    device = next(model.parameters()).device
    _, height, width = y.shape
    positions = [(top, left) for top in starts(height, patch)
                 for left in starts(width, patch)]
    if len(positions) != 25:
        raise ValueError("registered 512x512 validation geometry must contain 25 patches")

    load_start = time.monotonic()
    source = y.contiguous()
    if pin_memory and device.type == "cuda" and not source.is_pinned():
        source = source.pin_memory()
    _sync(device)
    transfer_start = time.monotonic()
    source_device = source.to(device, non_blocking=pin_memory and device.type == "cuda")
    output = torch.zeros((1, height, width), device=device)
    total = torch.zeros_like(output)
    axis = torch.hann_window(patch, periodic=False, device=device).clamp_min(1e-3)
    window = (axis[:, None] * axis[None]).clamp_min(1e-6)
    _sync(device)
    forward_start = time.monotonic()

    for offset in range(0, len(positions), patch_batch):
        group = positions[offset:offset + patch_batch]
        crops = torch.stack(
            [source_device[:, top:top + patch, left:left + patch]
             for top, left in group],
            dim=0,
        )
        if graph_cache is not None:
            if patch_batch != 1 or device.type != "cuda":
                raise ValueError("CUDA Graph is supported only for CUDA patch_batch=1")
            if "runner" not in graph_cache:
                graph_cache["runner"] = FixedShapeGraphRunner(
                    model, crops, alpha_a, alpha_b, alpha_extension
                )
            predictions = graph_cache["runner"](crops).float()
        else:
            predictions = model(crops, alpha_a=alpha_a, alpha_b=alpha_b,
                                alpha_extension=alpha_extension).float()
        for prediction, (top, left) in zip(predictions, group):
            output[:, top:top + patch, left:left + patch].add_(prediction * window)
            total[:, top:top + patch, left:left + patch].add_(window)

    _sync(device)
    d2h_start = time.monotonic()
    prediction = (output / total).clamp(-1024, 3072).cpu()
    _sync(device)
    end = time.monotonic()
    return prediction, {
        "load_sec": transfer_start - load_start,
        "h2d_setup_sec": forward_start - transfer_start,
        "model_stitch_sec": d2h_start - forward_start,
        "d2h_sec": end - d2h_start,
        "total_infer_sec": end - load_start,
    }


@torch.no_grad()
def evaluate_v29(
    model: V29CTSystem,
    dataset,
    *,
    alpha_a: float = 1.0,
    alpha_b: float = 1.0,
    alpha_extension: float = 1.0,
    indices: Iterable[int] | None = None,
    patch_batch: int = 1,
    pin_memory: bool = False,
    prediction_callback: Callable[[int, torch.Tensor], None] | None = None,
    metric_workers: int = 0,
    use_cuda_graph: bool = False,
) -> dict:
    was_training = model.training
    model.eval()
    ordered = list(range(len(dataset))) if indices is None else [int(index) for index in indices]
    rows: list[dict] = []
    if metric_workers not in {0, 1, 2}:
        raise ValueError("metric_workers must be 0/1/2")
    totals = {key: 0.0 for key in
              ("load_sec", "h2d_setup_sec", "model_stitch_sec", "d2h_sec",
               "metric_sec", "total_infer_sec")}
    started = time.monotonic()
    graph_cache = {} if use_cuda_graph else None
    executor = ThreadPoolExecutor(max_workers=metric_workers) if metric_workers else None
    pending: list[tuple[Future, dict, dict]] = []

    def calculate(prediction: torch.Tensor, target: torch.Tensor) -> tuple[dict, float]:
        metric_started = time.monotonic()
        metrics = patch_metrics(
            ((prediction + 1024) / 4096)[None],
            ((target + 1024) / 4096)[None],
        )
        metrics["psnr_db"] = metrics.pop("psnr_hu")
        metrics["ssim"] = metrics.pop("ssim_hu")
        metrics.update(region_metrics(prediction, target))
        return metrics, time.monotonic() - metric_started

    def finish(item: tuple[Future, dict, dict]) -> None:
        future, metadata, timing = item
        metrics, metric_sec = future.result()
        timing["metric_sec"] = metric_sec
        for key in totals:
            totals[key] += float(timing.get(key, 0.0))
        rows.append({**metadata, **metrics})

    try:
        for index in ordered:
            load_started = time.monotonic()
            row = dataset[index]
            y, target = row["y"], row["target_hu"]
            dataset_load_sec = time.monotonic() - load_started
            prediction, timing = infer_image_v29(
                model, y, alpha_a=alpha_a, alpha_b=alpha_b,
                alpha_extension=alpha_extension,
                patch_batch=patch_batch, pin_memory=pin_memory
                , graph_cache=graph_cache
            )
            if prediction_callback is not None:
                prediction_callback(index, prediction)
            timing["load_sec"] += dataset_load_sec
            metadata = {
                "original_index": index,
                "patient_id": row["patient_id"],
                "series_id": row["series_id"],
                "slice_id": row["slice_id"],
            }
            if executor is None:
                future: Future = Future()
                future.set_result(calculate(prediction, target))
            else:
                future = executor.submit(calculate, prediction, target.clone())
            pending.append((future, metadata, timing))
            if len(pending) >= max(1, metric_workers):
                finish(pending.pop(0))
        while pending:
            finish(pending.pop(0))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        model.train(was_training)
    rows.sort(key=lambda row: ordered.index(row["original_index"]))
    result = summarize([{key: value for key, value in row.items()
                         if key != "original_index"} for row in rows])
    result.update({
        "evaluation_role": (
            "deployment_validation"
            if alpha_a == 1.0 and alpha_b == 1.0 and alpha_extension == 1.0
            else "eval_only_probe"
        ),
        "alpha_a": float(alpha_a),
        "alpha_b": float(alpha_b),
        "alpha_extension": float(alpha_extension),
        "patch_batch": int(patch_batch),
        "pin_memory": bool(pin_memory),
        "use_cuda_graph": bool(use_cuda_graph),
        "regions": "target>=200HU; top10percent gradients inside target>=-900HU",
        "rows": rows,
        "timing": {**totals, "wall_time_sec": time.monotonic() - started},
    })
    return result


def fixed_screen_indices(length: int, count: int = 32) -> list[int]:
    if length < count:
        raise ValueError("validation set too small for fixed screen")
    values = np.linspace(0, length - 1, count).round().astype(int).tolist()
    if len(set(values)) != count:
        raise RuntimeError("screen indices are not unique")
    return values


def compare_validation(reference: dict, candidate: dict) -> dict:
    if [row["original_index"] for row in reference["rows"]] != [
            row["original_index"] for row in candidate["rows"]]:
        raise ValueError("sample order mismatch")
    keys = ("psnr_db", "ssim", "mae_hu", "signed_bias_hu",
            "high_density_mae_hu", "high_density_bias_hu",
            "high_gradient_mae_hu")
    differences = {
        key: abs(float(candidate["metrics"][key]) - float(reference["metrics"][key]))
        for key in keys
    }
    limits = {
        "psnr_db": 1e-6,
        "ssim": 1e-8,
        "mae_hu": 1e-5,
        "signed_bias_hu": 1e-5,
        "high_density_mae_hu": 1e-5,
        "high_density_bias_hu": 1e-5,
        "high_gradient_mae_hu": 1e-5,
    }
    return {
        "sample_identity_passed": True,
        "aggregation_passed": reference["aggregation"] == candidate["aggregation"],
        "metric_differences": differences,
        "metric_limits": limits,
        "metrics_passed": all(differences[key] <= limits[key] for key in keys),
    }
