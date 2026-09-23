"""Reproducible single-GPU runner for the resolved RoDiff-CT v29 contract."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ct_prior.data import BatchStream, MayoFive
from ct_prior.gates import selection_key

from .audit_utils import atomic_json
from .contracts import validate_run_contract
from .model import (
    FORMAL_VARIANTS, build_v29_objective, create_v29_optimizer,
    is_extension_key, state_sha256,
)
from .validation import evaluate_v29

CHECKPOINT_SCHEMA = "rodiff_ct_v29_training_v1"
DEPLOYMENT_SCHEMA = "rodiff_ct_v29_deployment_v1"
IDENTITY_FIELDS = (
    "code_sha256", "manifest_sha256", "lr_trace_sha256",
    "historical_b0_curve_sha256", "resolved_contract_sha256",
    "native_source_sha256", "objective_sha256", "optimizer_config_sha256",
    "scheduler_config_sha256", "data_order_contract_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         default=str).encode()).hexdigest()


def torch_payload_sha256(value) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256()
    src_root = Path(__file__).resolve().parents[1]
    paths = sorted(list((src_root / "ct_v29_prod").glob("*.py")) +
                   list((src_root / "ct_v29").glob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(src_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_config(path: str | Path, *, mode: str) -> tuple[dict, list[float], dict[int, dict]]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_run_contract(config, mode=mode)
    for key in ("manifest_path", "lr_trace_path", "historical_b0_curve_path",
                "resolved_contract_path"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"missing {key}: {config[key]}")
    trace_rows = [json.loads(line) for line in Path(config["lr_trace_path"])
                  .read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(trace_rows) != 8200:
        raise ValueError("B0 LR trace must contain exactly 8200 updates")
    for index, row in enumerate(trace_rows, 1):
        if int(row["global_step"]) != index or int(row["epoch"]) != (index - 1) // 41 + 1:
            raise ValueError("B0 LR trace identity/order mismatch")
    b0_payload = json.loads(Path(config["historical_b0_curve_path"]).read_text(encoding="utf-8"))
    b0_rows = b0_payload["validation_curve"] if isinstance(b0_payload, dict) else b0_payload
    b0_curve = {int(row["epoch"]): row for row in b0_rows}
    expected = set(range(10, 201, 10))
    if set(b0_curve) != expected:
        raise ValueError("historical B0 curve must cover epochs 10..200")
    config["manifest_sha256"] = sha256_file(config["manifest_path"])
    config["lr_trace_sha256"] = sha256_file(config["lr_trace_path"])
    config["historical_b0_curve_sha256"] = sha256_file(config["historical_b0_curve_path"])
    config["resolved_contract_sha256"] = sha256_file(config["resolved_contract_path"])
    config["code_sha256"] = source_sha256()
    return config, [float(row["lr"]) for row in trace_rows], b0_curve


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_sha256(state: dict | None = None) -> str:
    state = rng_state() if state is None else state
    numpy_state = state["numpy"]
    digest = hashlib.sha256(
        repr((state["python"], numpy_state[0], numpy_state[2:])).encode()
    )
    digest.update(numpy_state[1].tobytes())
    digest.update(state["torch"].cpu().numpy().tobytes())
    for tensor in state["cuda"]:
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def make_runtime(config: dict, device: torch.device):
    seed_all(int(config["seed"]))
    objective = build_v29_objective(config)
    native_sha = state_sha256(objective.system.state_dict(), native_only=True)
    objective = objective.to(device)
    optimizer, coverage = create_v29_optimizer(objective, config)
    return objective, optimizer, native_sha, coverage


def train_update(objective, optimizer, stream, config, device, lr: float) -> dict:
    for group in optimizer.param_groups:
        group["lr"] = lr
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    losses: list[float] = []
    for _ in range(int(config["accumulation"])):
        y, target = stream.next()
        result = objective(y.to(device, non_blocking=True), target.to(device, non_blocking=True))
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        (loss / int(config["accumulation"])).backward()
        losses.append(float(loss.detach()))
    extension = [(name, parameter) for name, parameter in objective.named_parameters()
                 if is_extension_key(name)]
    monitored = extension or [(name, parameter) for name, parameter in objective.named_parameters()
                              if parameter.requires_grad and name.endswith("output.weight")]
    if not monitored:
        raise RuntimeError("no trainable sentinel parameter found")
    grad_norm = sum(float(parameter.grad.detach().float().square().sum())
                    for _, parameter in monitored if parameter.grad is not None) ** 0.5
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("training gradient is absent")
    before = {name: parameter.detach().clone() for name, parameter in monitored}
    optimizer.step()
    changed = [name for name, parameter in monitored
               if not torch.equal(before[name], parameter.detach())]
    if not changed:
        raise RuntimeError("trainable parameters did not update")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.monotonic() - started
    return {
        "loss": float(np.mean(losses)), "extension_grad_norm": grad_norm,
        "changed_extension_parameters": changed, "update_time_sec": seconds,
        "samples_per_sec": int(config["effective_batch"]) / max(seconds, 1e-9),
    }


def checkpoint_payload(objective, optimizer, stream, config, epoch, global_step,
                       best_order, validation_result=None, protocol_migrations=None) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA, "variant": config["variant"],
        "config": config, "config_sha256": json_sha256(config),
        "epoch": int(epoch), "global_step": int(global_step),
        "objective": objective.state_dict(), "optimizer": optimizer.state_dict(),
        "stream": stream.state_dict(), "rng": rng_state(),
        "lr_scheduler": {
            "kind": "frozen_8200_step_trace",
            "completed_global_step": int(global_step),
            "next_trace_index": int(global_step),
        },
        "accumulation_cursor": 0,
        "best_order": list(best_order) if best_order is not None else None,
        "validation_result": validation_result,
        "protocol_migrations": list(protocol_migrations or []),
        "identity": {key: config[key] for key in IDENTITY_FIELDS},
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(path: Path, objective, optimizer, stream, config,
                             epoch, global_step, best_order, validation_result=None,
                             protocol_migrations=None) -> None:
    atomic_torch_save(checkpoint_payload(objective, optimizer, stream, config, epoch,
                                         global_step, best_order, validation_result,
                                         protocol_migrations), path)


def export_deployment(path: Path, objective, config: dict, training_checkpoint: Path) -> None:
    atomic_torch_save({
        "schema": DEPLOYMENT_SCHEMA, "variant": config["variant"],
        "system": objective.system.state_dict(), "config": config,
        "training_checkpoint_sha256": sha256_file(training_checkpoint),
        "teacher_included": False,
    }, path)


def load_resume(path: Path, objective, optimizer, stream, config: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid v29 checkpoint schema; v27/B0 warm-start is forbidden")
    if checkpoint["variant"] != config["variant"]:
        raise ValueError("cross-variant resume is forbidden")
    if checkpoint["config_sha256"] != json_sha256(checkpoint["config"]):
        raise ValueError("checkpoint config is corrupt")
    if checkpoint["config_sha256"] != json_sha256(config):
        raise ValueError("resume configuration mismatch")
    if set(checkpoint.get("identity", {})) != set(IDENTITY_FIELDS):
        raise ValueError("checkpoint resume identity fields are incomplete")
    if checkpoint.get("accumulation_cursor") != 0:
        raise ValueError("checkpoint is not at an optimizer-update boundary")
    if checkpoint.get("lr_scheduler", {}).get("next_trace_index") != checkpoint["global_step"]:
        raise ValueError("checkpoint LR trace cursor mismatch")
    for key, value in checkpoint["identity"].items():
        if config.get(key) != value:
            raise ValueError(f"resume identity mismatch: {key}")
    objective.load_state_dict(checkpoint["objective"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    stream.epoch = int(checkpoint["stream"]["epoch"])
    stream.cursor = int(checkpoint["stream"]["cursor"])
    restore_rng(checkpoint["rng"])
    return checkpoint


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def delta_against_b0(metrics: dict, reference: dict) -> dict:
    keys = ("psnr_db", "ssim", "mae_hu", "signed_bias_hu",
            "high_density_mae_hu", "high_density_bias_hu",
            "high_gradient_mae_hu")
    return {f"delta_{key}": float(metrics[key] - reference[key]) for key in keys}


def dataset_config(config: dict) -> dict:
    return {"seed": config["seed"], "context_mode": config.get("context_mode", "real_five"),
            "paths": {"manifest": config["manifest_path"]}}


def evaluate_checkpoint(objective, validation, config, *,
                        alpha_a: float = 1.0, alpha_b: float = 1.0,
                        alpha_extension: float = 1.0,
                        indices=None) -> dict:
    return evaluate_v29(
        objective.system, validation, alpha_a=alpha_a, alpha_b=alpha_b,
        alpha_extension=alpha_extension,
        indices=indices,
        patch_batch=int(config["validation_patch_batch"]),
        pin_memory=bool(config["validation_pin_memory"]),
        metric_workers=int(config["validation_metric_workers"]),
        use_cuda_graph=bool(config["validation_cuda_graph"]),
    )


def run_probe(checkpoint_path: Path, objective, optimizer, stream, best_order,
              validation, config, epoch: int, global_step: int) -> dict:
    state_before = state_sha256(objective.state_dict())
    rng_before = rng_sha256()
    optimizer_before = torch_payload_sha256(optimizer.state_dict())
    stream_before = json_sha256(stream.state_dict())
    best_before = json_sha256(list(best_order) if best_order is not None else None)
    optimizer_steps_before = global_step
    probes = []
    variant = config["variant"]
    settings = [("on", 1.0, 1.0, 1.0)]
    if variant.startswith("A"):
        settings.extend((("a_half", 0.5, 1.0, 1.0),
                         ("a_raw", 0.0, 1.0, 1.0)))
    if variant.startswith("B") or variant.startswith("AB"):
        settings.extend((("b_half", 1.0, 0.5, 1.0),
                         ("b_mean", 1.0, 0.0, 1.0)))
    settings.append(("extension_off", 1.0, 1.0, 0.0))
    for label, alpha_a, alpha_b, alpha_extension in settings:
        probes.append({"label": label, "alpha_a": alpha_a, "alpha_b": alpha_b,
                       "alpha_extension": alpha_extension,
                       "evaluation_role": "eval_only",
                       "eligible_for_checkpoint_selection": False,
                       "result": evaluate_checkpoint(
                           objective, validation, config,
                           alpha_a=alpha_a, alpha_b=alpha_b,
                           alpha_extension=alpha_extension,
                       )})
    state_after = state_sha256(objective.state_dict())
    rng_after = rng_sha256()
    optimizer_after = torch_payload_sha256(optimizer.state_dict())
    stream_after = json_sha256(stream.state_dict())
    best_after = json_sha256(list(best_order) if best_order is not None else None)
    if any((state_before != state_after, rng_before != rng_after,
            optimizer_before != optimizer_after, stream_before != stream_after,
            best_before != best_after)):
        raise RuntimeError("probe mutated training state")
    return {
        "schema": "rodiff_ct_v29_probe_v1", "evaluation_role": "eval_only",
        "variant": config["variant"], "epoch": epoch, "global_step": global_step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_sha256": state_before, "manifest_sha256": config["manifest_sha256"],
        "run_id": Path(config["output_dir"]).name,
        "metric_contract_sha256": config["objective_sha256"],
        "data_order_contract_sha256": config["data_order_contract_sha256"],
        "code_sha256": config["code_sha256"],
        "ema_identity": "NOT_USED_BY_NATIVE_RUNNER",
        "optimizer_step_before": optimizer_steps_before,
        "optimizer_step_after": global_step, "rng_sha256_before": rng_before,
        "rng_sha256_after": rng_after,
        "optimizer_sha256_before": optimizer_before,
        "optimizer_sha256_after": optimizer_after,
        "stream_sha256_before": stream_before,
        "stream_sha256_after": stream_after,
        "best_order_sha256_before": best_before,
        "best_order_sha256_after": best_after,
        "probes": probes,
    }


def run_probe_process(config_path: Path, checkpoint_path: Path, output_path: Path,
                      epoch: int, global_step: int) -> dict:
    """Run a retained-epoch probe in a disposable CUDA process.

    A probe is evaluation-only and must never own the lifetime of the formal
    trainer.  Process isolation also prevents a failed CUDA graph/capture from
    poisoning the training process's CUDA context.
    """
    command = [
        sys.executable, "-m", "ct_v29_prod.training",
        "--config", str(config_path), "--mode", "probe",
        "--probe-checkpoint", str(checkpoint_path),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=4 * 60 * 60, env=os.environ.copy(),
        )
        returncode = int(completed.returncode)
        stdout_tail = completed.stdout[-4000:]
        stderr_tail = completed.stderr[-12000:]
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout_tail = (error.stdout or "")[-4000:]
        stderr_tail = (error.stderr or "")[-12000:]
    success = returncode == 0 and output_path.is_file()
    status = {
        "schema": "rodiff_ct_v29_probe_supervisor_v1",
        "epoch": int(epoch), "global_step": int(global_step),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_output_path": str(output_path),
        "process_isolated": True,
        "cuda_graph_policy": "loaded_from_probe_cuda_graph_config",
        "returncode": returncode,
        "success": success,
        "training_continued": not success,
        "wall_time_sec": time.monotonic() - started,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    atomic_json(output_path.with_suffix(".supervisor.json"), status)
    if not success:
        atomic_json(output_path.with_suffix(".failure.json"), status)
    return status


def probe_from_checkpoint(config: dict, checkpoint_path: Path) -> Path:
    """Execute one isolated probe; failures are handled by the parent trainer."""
    device = torch.device("cuda")
    output = Path(config["output_dir"])
    objective, optimizer, _, _ = make_runtime(config, device)
    train = MayoFive(dataset_config(config),
                     "train", output / "pixel_access.probe_train.jsonl")
    validation = MayoFive(
        dataset_config(config),
        "validation", output / "pixel_access.probe_validation.jsonl",
    )
    stream = BatchStream(train, config["seed"], batch_size=config["micro_batch"])
    restored = load_resume(checkpoint_path, objective, optimizer, stream, config)
    epoch = int(restored["epoch"])
    global_step = int(restored["global_step"])
    best_order = tuple(restored["best_order"]) if restored["best_order"] else None
    # v29 records tensor diagnostics with .item() inside forward.  CUDA Graph
    # capture forbids that operation, so probes use eager CUDA plus the tuned
    # patch batch instead of risking the formal trainer for an invalid speedup.
    probe_config = dict(config)
    probe_config["validation_cuda_graph"] = bool(config.get("probe_cuda_graph", False))
    report = run_probe(
        checkpoint_path, objective, optimizer, stream, best_order,
        validation, probe_config, epoch, global_step,
    )
    report["execution"] = {
        "process_isolated": True,
        "device": str(device),
        "cuda_graph": bool(probe_config["validation_cuda_graph"]),
        "patch_batch": int(probe_config["validation_patch_batch"]),
        "internal_tensor_diagnostics": False,
    }
    output_path = output / f"probe_epoch_{epoch:03d}.json"
    atomic_json(output_path, report)
    return output_path


def smoke(config: dict, lr_trace: list[float]) -> Path:
    device = torch.device("cuda")
    output = Path(
        str(config["output_dir"])
        + "_smoke_"
        + config["code_sha256"][:12]
        + "_"
        + json_sha256(config)[:12]
    )
    output.mkdir(parents=True, exist_ok=False)
    objective, optimizer, native_sha, coverage = make_runtime(config, device)
    train = MayoFive(dataset_config(config),
                     "train", output / "pixel_access.jsonl")
    stream = BatchStream(train, config["seed"], batch_size=config["micro_batch"])
    writer = SummaryWriter(str(output / "tensorboard"))
    try:
        first = train_update(objective, optimizer, stream, config, device, lr_trace[0])
        writer.add_scalar("train/loss", first["loss"], 1)
        writer.add_scalar("train/lr", lr_trace[0], 1)
        writer.add_scalar("train/grad_norm", first["extension_grad_norm"], 1)
        writer.add_scalar("train/optimizer_step", 1, 1)
        writer.add_scalar("train/throughput_samples_per_sec", first["samples_per_sec"], 1)
        writer.add_scalar("system/gpu_memory_peak_mb",
                          torch.cuda.max_memory_allocated(device) / 1048576, 1)
        checkpoint = output / "resume.pt"
        save_training_checkpoint(checkpoint, objective, optimizer, stream, config, 0, 1, None)
        objective2, optimizer2, _, _ = make_runtime(config, device)
        stream2 = BatchStream(train, config["seed"], batch_size=config["micro_batch"])
        restored = load_resume(checkpoint, objective2, optimizer2, stream2, config)
        second = train_update(objective2, optimizer2, stream2, config, device, lr_trace[1])
        writer.add_scalar("train/loss", second["loss"], 2)
        writer.add_scalar("train/lr", lr_trace[1], 2)
        writer.add_scalar("train/grad_norm", second["extension_grad_norm"], 2)
        writer.add_scalar("train/optimizer_step", 2, 2)
        writer.add_scalar("train/throughput_samples_per_sec", second["samples_per_sec"], 2)
        writer.add_scalar("system/gpu_memory_peak_mb",
                          torch.cuda.max_memory_allocated(device) / 1048576, 2)
        writer.flush()
    finally:
        writer.close()
    result = {
        "schema": "rodiff_ct_v29_smoke_v1", "smoke_pass": True,
        "global_step_before_resume": restored["global_step"], "global_step_after": 2,
        "lr_step1": lr_trace[0], "lr_step2": lr_trace[1],
        "first_update": first, "second_update": second,
        "native_initial_state_sha256": native_sha, "optimizer_coverage": coverage,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return output / "result.json"


def native_preflight(config: dict, *, use_cuda: bool) -> dict:
    device = torch.device("cuda" if use_cuda else "cpu")
    variant = config["variant"]
    candidate, _, native_sha, coverage = make_runtime(config, device)
    baseline_config = dict(config)
    baseline_config["variant"] = "B0"
    baseline, _, baseline_native_sha, _ = make_runtime(baseline_config, device)
    sizes = ((32, 32), (128, 128), (128, 160))
    if use_cuda:
        sizes = sizes + ((512, 512),)
    rows = []
    for height, width in sizes:
        generator = torch.Generator(device="cpu").manual_seed(280000 + height + width)
        value = torch.randn((1, 5, height, width), generator=generator).to(device)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.no_grad():
            expected = baseline.system(value)
            observed = candidate.system(value)
        if use_cuda:
            torch.cuda.synchronize(device)
        rows.append({
            "shape": list(value.shape),
            "exact_initial_parity": torch.equal(expected, observed),
            "max_abs_hu_difference": float((expected - observed).abs().max()),
            "finite": bool(torch.isfinite(observed).all()),
            "synchronized_forward_wall_sec": time.monotonic() - started,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1048576
                if use_cuda else None
            ),
        })
    active_512 = None
    backward = None
    if use_cuda:
        if candidate.system.restorer.carrier is not None:
            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(290512)
                weight = torch.randn(
                    candidate.system.restorer.carrier.output.weight.shape,
                    generator=generator,
                ).mul_(0.01).to(device)
                candidate.system.restorer.carrier.output.weight.copy_(weight)
                value_active = torch.randn((1, 5, 512, 512), device=device)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                active_started = time.monotonic()
                active_output = candidate.system(value_active)
                torch.cuda.synchronize(device)
                active_512 = {
                    "finite": bool(torch.isfinite(active_output).all()),
                    "nonzero_extension": bool(torch.count_nonzero(
                        active_output - baseline.system(value_active)
                    )),
                    "synchronized_forward_wall_sec": time.monotonic() - active_started,
                    "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1048576,
                }
        # The full-resolution FP32 graph is larger than one A40 when every
        # saved activation stays resident on device.  This is a preflight-only
        # finite-gradient check, so offload saved tensors to pinned host memory
        # without changing the production train path or numerical precision.
        del expected, observed
        del baseline
        torch.cuda.empty_cache()
        value = torch.randn((1, 5, 512, 512), device=device, requires_grad=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            loss = candidate.system(value).float().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        backward = {
            "finite_input_gradient": bool(value.grad is not None and torch.isfinite(value.grad).all()),
            "synchronized_backward_wall_sec": time.monotonic() - started,
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1048576,
        }
    extension_count = coverage["extension_parameter_count"]
    expected_count = {"B0": 0, "B05center": 0, "A": 8082, "B": 13545, "AB": 15435}.get(variant)
    if expected_count is not None and extension_count != expected_count:
        raise RuntimeError(f"extension parameter count {extension_count} != {expected_count}")
    if not all(row["exact_initial_parity"] and row["finite"] for row in rows):
        raise RuntimeError("native initial parity preflight failed")
    return {
        "schema": "rodiff_ct_v29_native_preflight_v1",
        "variant": variant,
        "device": str(device),
        "native_initial_sha256": native_sha,
        "baseline_native_initial_sha256": baseline_native_sha,
        "optimizer_coverage": coverage,
        "shapes": rows,
        "active_512": active_512,
        "backward_512": backward,
    }


def formal(config: dict, lr_trace: list[float], b0_curve: dict[int, dict], *,
           resume: bool, config_path: Path) -> None:
    device = torch.device("cuda")
    output = Path(config["output_dir"])
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
    objective, optimizer, native_sha, coverage = make_runtime(config, device)
    train = MayoFive(dataset_config(config),
                     "train", output / "pixel_access.train.jsonl")
    validation = MayoFive(dataset_config(config),
                          "validation", output / "pixel_access.validation.jsonl")
    stream = BatchStream(train, config["seed"], batch_size=config["micro_batch"])
    global_step, start_epoch, best_order = 0, 1, None
    protocol_migrations = []
    if resume:
        restored = load_resume(output / "last.pt", objective, optimizer, stream, config)
        global_step = int(restored["global_step"])
        start_epoch = int(restored["epoch"]) + 1
        best_order = tuple(restored["best_order"]) if restored["best_order"] else None
        protocol_migrations = list(restored.get("protocol_migrations", []))
    contract = {
        "schema": "rodiff_ct_v29_run_contract_v1", "variant": config["variant"],
        "fresh_initialization": not resume, "native_initial_state_sha256": native_sha,
        "optimizer_coverage": coverage,
        **{key: config[key] for key in (
            "target_instance", "code_sha256", "manifest_sha256", "lr_trace_sha256",
            "historical_b0_curve_sha256", "resolved_contract_sha256")},
        "validation_interval_epochs": int(config["validation_interval_epochs"]),
        "protocol_migrations": protocol_migrations,
        **config["historical_reference_b0"],
    }
    if not resume:
        atomic_json(output / "contract.json", contract)
    else:
        atomic_json(output / "contract_current.json", contract)
    log_path = output / "training.jsonl"
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=global_step or None)
    steps_per_epoch = len(train) // int(config["effective_batch"])
    if steps_per_epoch != 41:
        raise ValueError(f"expected 41 optimizer updates/epoch, got {steps_per_epoch}")
    recent_epoch_seconds: list[float] = []
    try:
        for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
            epoch_started = time.monotonic()
            update_seconds = []
            for update_index in range(steps_per_epoch):
                result = train_update(objective, optimizer, stream, config, device,
                                      lr_trace[global_step])
                global_step += 1
                update_seconds.append(result["update_time_sec"])
                row = {
                    "epoch": epoch, "update_in_epoch": update_index + 1,
                    "global_step": global_step, "train/loss": result["loss"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": result["extension_grad_norm"],
                    "train/optimizer_step": global_step,
                    "train/update_time_sec": result["update_time_sec"],
                    "train/throughput_samples_per_sec": result["samples_per_sec"],
                    "system/gpu_memory_peak_mb": (
                        torch.cuda.max_memory_allocated(device) / 1048576),
                }
                append_jsonl(log_path, row)
                for key, value in row.items():
                    if key.startswith(("train/", "system/")):
                        writer.add_scalar(key, value, global_step)
            train_seconds = sum(update_seconds)
            save_training_checkpoint(output / "last.pt", objective, optimizer, stream,
                                     config, epoch, global_step, best_order,
                                     protocol_migrations=protocol_migrations)
            retain_epochs = set(map(int, config.get("retain_epochs", [40, 90, 130])))
            probe_epochs = set(map(int, config.get("probe_epochs", [40, 90, 130])))
            if epoch in retain_epochs:
                retained = output / f"checkpoint_epoch_{epoch:03d}.pt"
                save_training_checkpoint(retained, objective, optimizer, stream,
                                         config, epoch, global_step, best_order,
                                         protocol_migrations=protocol_migrations)
            validation_seconds = 0.0
            if epoch % int(config["validation_interval_epochs"]) == 0:
                validation_started = time.monotonic()
                main = evaluate_checkpoint(objective, validation, config)
                validation_seconds = time.monotonic() - validation_started
                reference = b0_curve[epoch]
                delta = delta_against_b0(main["metrics"], reference)
                main.update({
                    "schema": "rodiff_ct_v29_validation_v1", "epoch": epoch,
                    "global_step": global_step,
                    "evaluation_role": "formal_validation_historical_reference",
                    "historical_reference_b0": reference,
                    **config["historical_reference_b0"], **delta,
                })
                atomic_json(output / f"validation_epoch_{epoch:03d}.json", main)
                current_order = selection_key(main)
                if best_order is None or current_order > best_order:
                    best_order = current_order
                    save_training_checkpoint(output / "best.pt", objective, optimizer, stream,
                                             config, epoch, global_step, best_order, main,
                                             protocol_migrations)
                    export_deployment(output / "best_deployment.pt", objective, config,
                                      output / "best.pt")
                save_training_checkpoint(output / "last.pt", objective, optimizer, stream,
                                         config, epoch, global_step, best_order, main,
                                         protocol_migrations)
                for key, value in main["metrics"].items():
                    writer.add_scalar("val/" + key, value, epoch)
                for key, value in delta.items():
                    writer.add_scalar("compare/" + key, value, epoch)
                for key, value in main["timing"].items():
                    writer.add_scalar("val_timing/" + key, value, epoch)
                if epoch in probe_epochs:
                    retained = output / f"checkpoint_epoch_{epoch:03d}.pt"
                    probe_output = output / f"probe_epoch_{epoch:03d}.json"
                    probe_status = run_probe_process(
                        config_path, retained, probe_output, epoch, global_step,
                    )
                    writer.add_scalar("probe/success", float(probe_status["success"]), epoch)
            epoch_seconds = time.monotonic() - epoch_started
            recent_epoch_seconds.append(epoch_seconds)
            recent_epoch_seconds = recent_epoch_seconds[-10:]
            remaining = int(config["max_epochs"]) - epoch
            eta_seconds = float(np.mean(recent_epoch_seconds) * remaining)
            cycle = {
                "epoch": epoch, "global_step": global_step,
                "train_wall_time_sec": train_seconds,
                "validation_wall_time_sec": validation_seconds,
                "epoch_wall_time_sec": epoch_seconds,
                "optimizer_updates": steps_per_epoch,
                "mean_update_time_sec": float(np.mean(update_seconds)),
                "epoch_throughput_samples_per_sec": (
                    steps_per_epoch * int(config["effective_batch"]) / max(train_seconds, 1e-9)),
                "estimated_remaining_seconds": eta_seconds,
                "eta_status": ("OBSERVED_COMPLETE_CYCLE" if
                               epoch >= int(config["validation_interval_epochs"])
                               else "ETA_UNCERTAIN"),
            }
            atomic_json(output / "eta.json", cycle)
            append_jsonl(output / "epoch_summary.jsonl", cycle)
            writer.add_scalar("train/epoch_time_sec", train_seconds, epoch)
            writer.add_scalar("system/epoch_wall_time_sec", epoch_seconds, epoch)
            writer.flush()
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True,
                        choices=("native-preflight", "smoke", "profile", "formal", "resume", "probe"))
    parser.add_argument("--probe-checkpoint")
    args = parser.parse_args()
    config, lr_trace, b0_curve = load_config(args.config, mode=args.mode)
    if args.mode == "native-preflight":
        print(json.dumps(native_preflight(config, use_cuda=False), ensure_ascii=False))
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    observed_uuid = str(torch.cuda.get_device_properties(0).uuid)
    expected_uuid = str(config["authorized_gpu_uuid"]).removeprefix("GPU-")
    if observed_uuid != expected_uuid:
        raise RuntimeError("visible CUDA device UUID is not the authorized GPU")
    if args.mode == "profile":
        print(json.dumps(native_preflight(config, use_cuda=True), ensure_ascii=False))
    elif args.mode == "smoke":
        smoke(config, lr_trace)
    elif args.mode == "probe":
        if not args.probe_checkpoint:
            parser.error("--probe-checkpoint is required for probe mode")
        print(probe_from_checkpoint(config, Path(args.probe_checkpoint)))
    else:
        formal(config, lr_trace, b0_curve, resume=args.mode == "resume",
               config_path=Path(args.config).resolve())


if __name__ == "__main__":
    main()
