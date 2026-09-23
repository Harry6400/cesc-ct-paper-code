"""Two-rank formal v31 runner preserving global effective batch 32."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import itertools
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cesc.audit import append_jsonl, file_sha256, implementation_sha256, module_sha256, write_json
from cesc.checkpoint import load_checkpoint, save_checkpoint
from cesc.config import ModelConfig, config_hash
from cesc.evaluate import evaluate, evaluate_statistics
from cesc.losses import reconstruction_loss
from cesc.metrics import PatientMetrics, paired_delta
from cesc.model import CESC
from cesc.runner import import_bridge, native_gate
from cesc.statistics import statistics_loss


class StageObjective(nn.Module):
    def __init__(self, core: CESC):
        super().__init__()
        self.core = core

    def forward(self, center_hu, b0_hu, target_hu):
        if self.core.stage == "statistics":
            metric = self.core.statistic_metric(center_hu, b0_hu)
            loss, _ = statistics_loss(metric, (target_hu - b0_hu) / self.core.cfg.s_ct)
            return loss
        prediction = self.core(center_hu, b0_hu)
        return reconstruction_loss(prediction, target_hu, self.core.cfg.s_ct)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(dist.get_world_size())
    return result


def merge_validation_results(rank_results: list[dict], expected_count: int) -> dict:
    candidate = PatientMetrics()
    baseline = PatientMetrics()
    for result in rank_results:
        if not isinstance(result, dict):
            raise RuntimeError("missing rank-local validation result")
        for row in result.get("slice_metrics", []):
            candidate.add(row["patient_id"], row["slice_id"], row["candidate"])
            baseline.add(row["patient_id"], row["slice_id"], row["fixed_b0"])
    if len(candidate.rows) != expected_count or len(baseline.rows) != expected_count:
        raise RuntimeError(
            f"distributed validation reconstructed {len(candidate.rows)} slices; "
            f"expected {expected_count}"
        )
    return paired_delta(candidate, baseline)


def evaluate_distributed(model, bridge, device, rank: int, world: int) -> dict | None:
    total = len(bridge.validation_dataset)
    indices = list(range(rank, total, world))
    local = evaluate(model, bridge, device, indices=indices)
    gathered = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return merge_validation_results(gathered, total) if rank == 0 else None


def train_epoch(ddp, core, bridge, optimizer, epoch, accumulation, device, micro_batch,
                writer, training_log, global_step):
    ddp.train()
    optimizer.zero_grad(set_to_none=True)
    batches = iter(bridge.train_batches(epoch))
    total_loss = 0.0
    total_samples = 0
    epoch_started = time.monotonic()
    optimizer_steps = 0
    while True:
        group = list(itertools.islice(batches, accumulation))
        if not group:
            break
        if len(group) != accumulation:
            raise RuntimeError("rank-local microbatch count must divide accumulation exactly")
        update_started = time.monotonic()
        update_loss = torch.zeros((), device=device)
        samples = 0
        for index, batch in enumerate(group):
            batch.validate(micro_batch)
            batch = batch.to(device)
            context = ddp.no_sync() if index + 1 < len(group) else nullcontext()
            with context:
                loss = ddp(batch.center_hu, batch.b0_hu, batch.target_hu)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                (loss / accumulation).backward()
            update_loss += loss.detach()
            samples += len(batch.patient_ids)
        grad_sq = torch.zeros((), device=device)
        for parameter in core.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite training gradient")
                grad_sq += parameter.grad.detach().float().square().sum()
        grad_norm = grad_sq.sqrt()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        global_step += 1
        optimizer_steps += 1
        mean_loss = reduce_mean(update_loss / accumulation)
        mean_grad = reduce_mean(grad_norm)
        update_seconds = time.monotonic() - update_started
        total_loss += float(mean_loss) * samples * dist.get_world_size()
        total_samples += samples * dist.get_world_size()
        if dist.get_rank() == 0:
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train/loss": float(mean_loss),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": float(mean_grad),
                "train/optimizer_step": global_step,
                "train/throughput_samples_per_sec": (
                    micro_batch * accumulation * dist.get_world_size() / max(update_seconds, 1e-9)
                ),
                "train/update_time_sec": update_seconds,
                "system/gpu_memory_peak_mb": torch.cuda.max_memory_allocated(device) / 1048576,
            }
            append_jsonl(training_log, row)
            for key, value in row.items():
                if key.startswith(("train/", "system/")):
                    writer.add_scalar(key, value, global_step)
    bridge.assert_frozen_b0()
    return {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "optimizer_steps": optimizer_steps,
        "seconds": time.monotonic() - epoch_started,
        "global_step": global_step,
    }


def export_deployment(path: Path, core: CESC, provenance: dict, training_checkpoint: Path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "schema": "rodiff_ct_v31_deployment_v1",
        "variant": core.variant,
        "model": core.state_dict(),
        "provenance": provenance,
        "training_checkpoint_sha256": file_sha256(training_checkpoint),
        "native_b0_included": False,
    }, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bridge", default="bridges.native_e07_b0:NativeE07B0Bridge")
    parser.add_argument("--stats-checkpoint")
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--runtime-upgrade-audit")
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise RuntimeError("v31 E51 contract requires exactly two DDP ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = json.loads(Path(args.config).read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    bridge = import_bridge(args.bridge, config)
    contract = bridge.contract()
    native_gate(contract, config, True, args.preflight_report)
    cfg = ModelConfig(**config["model"])
    core = CESC(cfg, config["variant"]).to(device).set_stage(config["stage"])
    if config["stage"] == "correction":
        if not args.stats_checkpoint:
            raise RuntimeError("v31 Full correction requires the selected statistics checkpoint")
        payload = torch.load(args.stats_checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("provenance", {})
        if source.get("stage") != "statistics" or source.get("bridge_contract") != contract:
            raise RuntimeError("statistics checkpoint does not match fixed B0/data contract")
        state = {key.removeprefix("statistics."): value for key, value in payload["model"].items()
                 if key.startswith("statistics.")}
        core.statistics.load_state_dict(state, strict=True)
    objective = StageObjective(core)
    ddp = DDP(objective, device_ids=[local_rank], output_device=local_rank,
              broadcast_buffers=False, gradient_as_bucket_view=True)
    parameters = [parameter for parameter in core.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["lr"]),
                                  weight_decay=float(config.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=float(config["min_lr"])
    )
    current_provenance = {
        "bridge_contract": contract,
        "config_sha256": config_hash(config),
        "stage": config["stage"],
        "variant": config["variant"],
        "code_version": "0.31.1-rebuilt-production-ddp",
        "implementation_sha256": implementation_sha256(),
        "model_config": asdict(cfg),
        "stats_checkpoint_sha256": file_sha256(args.stats_checkpoint) if args.stats_checkpoint else None,
        "world_size": world,
        "global_effective_batch": int(config["effective_batch"]),
    }
    out = Path(args.out)
    provenance = current_provenance
    runtime_upgrade = None
    if args.resume:
        stored_provenance = json.loads((out / "provenance.json").read_text())
        scientific_keys = (
            "bridge_contract", "config_sha256", "stage", "variant", "model_config",
            "stats_checkpoint_sha256", "world_size", "global_effective_batch",
        )
        if any(stored_provenance.get(key) != current_provenance.get(key)
               for key in scientific_keys):
            raise RuntimeError("runtime upgrade changed frozen scientific provenance")
        if stored_provenance != current_provenance:
            if not args.runtime_upgrade_audit:
                raise RuntimeError("changed implementation requires --runtime-upgrade-audit")
            runtime_upgrade = json.loads(Path(args.runtime_upgrade_audit).read_text())
            if (runtime_upgrade.get("schema") != "rodiff_ct_runtime_upgrade_v1" or
                    runtime_upgrade.get("authorized") is not True or
                    runtime_upgrade.get("scientific_semantics_unchanged") is not True):
                raise RuntimeError("invalid runtime upgrade authorization")
            provenance = stored_provenance
    if rank == 0:
        if args.resume:
            if not out.is_dir():
                raise RuntimeError("resume directory is absent")
        else:
            out.mkdir(parents=True, exist_ok=False)
        write_json(out / "config.json", config)
        write_json(out / "provenance.json", provenance)
        if runtime_upgrade is not None:
            write_json(out / "runtime_upgrade_validation_ddp.json", {
                **runtime_upgrade,
                "previous_implementation_sha256": stored_provenance["implementation_sha256"],
                "active_runtime_implementation_sha256": current_provenance["implementation_sha256"],
            })
        write_json(out / "parameters.json", core.parameter_report())
        shutil.copy2(args.contract, out / "contract.json")
    dist.barrier()
    start, best, global_step = 1, (float("inf") if config["stage"] == "statistics" else -float("inf")), 0
    if args.resume:
        restored = load_checkpoint(args.resume, core, optimizer, scheduler, provenance)
        start, best = int(restored["epoch"]) + 1, restored["extra"]["best"]
        global_step = int(restored["extra"].get("global_step", 0))
    writer = SummaryWriter(str(out / "tensorboard")) if rank == 0 else None
    recent = []
    try:
        for epoch in range(start, int(config["epochs"]) + 1):
            train = train_epoch(ddp, core, bridge, optimizer, epoch,
                                int(config["accumulation"]), device,
                                int(config["micro_batch"]), writer,
                                out / "training.jsonl", global_step)
            global_step = train["global_step"]
            scheduler.step()
            dist.barrier()
            improved = False
            validation_seconds = 0.0
            result = None
            should_validate = epoch % int(config["val_interval"]) == 0 or epoch == int(config["epochs"])
            if should_validate:
                started = time.monotonic()
                if config["stage"] == "statistics":
                    if rank == 0:
                        result = evaluate_statistics(core, bridge, device)
                else:
                    result = evaluate_distributed(core, bridge, device, rank, world)
                validation_seconds = time.monotonic() - started
                if rank == 0:
                    score = (result["patient_equal_mean"]["working_nll"]
                             if config["stage"] == "statistics"
                             else result["candidate"]["patient_equal_mean"]["psnr_db"])
                    improved = score < best if config["stage"] == "statistics" else score > best
                    if improved:
                        best = score
                    write_json(out / f"validation_epoch_{epoch:03d}.json", result)
                    append_jsonl(out / "events.jsonl", {
                        "kind": "validation_main", "stage": config["stage"],
                        "variant": config["variant"], "epoch": epoch, "result": result,
                    })
                    if config["stage"] == "correction":
                        mapping = {
                            "psnr_db": "val/psnr_db", "ssim": "val/ssim",
                            "mae_hu": "val/mae_hu", "signed_bias_hu": "val/signed_bias_hu",
                            "high_density_mae_hu": "val/high_density_mae_hu",
                            "high_density_bias_hu": "val/high_density_bias_hu",
                            "high_gradient_mae_hu": "val/high_gradient_mae_hu",
                        }
                        for key, tag in mapping.items():
                            writer.add_scalar(tag, result["candidate"]["patient_equal_mean"][key], epoch)
            dist.barrier()
            if rank == 0:
                extra = {"best": best, "global_step": global_step,
                         "checkpoint_boundary": "end_of_epoch", "world_size": world}
                save_checkpoint(out / "last.pt", core, optimizer, scheduler, epoch, provenance, extra)
                if improved:
                    save_checkpoint(out / "best.pt", core, optimizer, scheduler, epoch, provenance, extra)
                    if config["stage"] == "correction":
                        export_deployment(out / "best_deployment.pt", core, provenance, out / "best.pt")
                recent.append(train["seconds"] + validation_seconds)
                recent = recent[-10:]
                remaining = int(config["epochs"]) - epoch
                summary = {
                    "epoch": epoch, "global_step": global_step,
                    "train_wall_time_sec": train["seconds"],
                    "validation_wall_time_sec": validation_seconds,
                    "epoch_wall_time_sec": train["seconds"] + validation_seconds,
                    "optimizer_updates": train["optimizer_steps"],
                    "estimated_remaining_seconds": float(np.mean(recent) * remaining),
                    "eta_status": "OBSERVED_COMPLETE_CYCLE" if should_validate else "ETA_UNCERTAIN",
                }
                append_jsonl(out / "epoch_summary.jsonl", summary)
                write_json(out / "eta.json", summary)
                writer.add_scalar("train/epoch_time_sec", train["seconds"], epoch)
                writer.flush()
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            dist.barrier()
            if should_validate:
                bridge.verify_b0_hash()
    finally:
        if writer:
            writer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
