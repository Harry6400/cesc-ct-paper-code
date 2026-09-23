"""Independent B0/D/G/M trainer. The legacy filename is retained for CLI compatibility."""
import argparse
import contextlib
import datetime
from dataclasses import asdict
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import export_deployment, read_checkpoint, restore_rng, save_checkpoint
from .config import fingerprint, load_config, sha256, verify_data_contract
from .data import BatchStream, make_dataset
from .evaluate import evaluate, summarize
from .gates import safety, selection_key
from .shared.early_stop import EarlyStopState
from .system import CTSystem, IndependentObjective


def usable_samples(dataset_size, effective_batch=32):
    """Samples in one complete training-ledger traversal; the tail is consistently dropped."""
    if dataset_size < effective_batch:
        raise ValueError("training ledger cannot form one effective batch")
    return dataset_size // effective_batch * effective_batch


def optimizer_updates_per_epoch(dataset_size, effective_batch=32):
    return usable_samples(dataset_size, effective_batch) // effective_batch


def completed_epochs(stream, dataset_size):
    return int(stream.epoch) + (int(stream.cursor) >= usable_samples(dataset_size))


def remaining_epoch_updates(stream, dataset_size, max_epochs):
    per_epoch = optimizer_updates_per_epoch(dataset_size)
    completed = completed_epochs(stream, dataset_size)
    if int(stream.cursor) >= usable_samples(dataset_size):
        return max(0, (int(max_epochs) - completed) * per_epoch)
    used = int(stream.cursor) // 32
    return max(0, (int(max_epochs) - int(stream.epoch) - 1) * per_epoch + per_epoch - used)


def restore_early_stop(payload):
    if not payload:
        return EarlyStopState()
    state = EarlyStopState(**payload)
    state.history = [(int(epoch), float(value)) for epoch, value in state.history]
    return state


def metric_view(result, reference):
    metrics, inputs = result["metrics"], reference["metrics"]
    return {"psnr_hu": float(metrics["psnr_db"]), "mae_hu": float(metrics["mae_hu"]),
            "ssim_hu": float(metrics["ssim"]), "input_psnr_hu": float(inputs["psnr_db"]),
            "input_mae_hu": float(inputs["mae_hu"]), "input_ssim_hu": float(inputs["ssim"])}


def _append_jsonl(path, payload):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _validate(system, validation, rank, world_size, config):
    indices = list(range(len(validation)))
    local = evaluate(system, validation, "whole", indices=indices[rank::world_size])
    if world_size == 1:
        return local
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local["rows"], gathered, dst=0)
    if rank != 0:
        return None
    rows = [row for shard in gathered for row in shard]
    result = summarize(rows)
    result.update({"kind": "whole", "evaluation_role": "deployment_validation",
                   "regions": local["regions"], "rows": rows})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--safety-reference", required=True)
    parser.add_argument("--gpu-authorized", action="store_true")
    args = parser.parse_args()
    if not args.gpu_authorized:
        parser.error("separate GPU run authorization required")
    config = load_config(args.config)
    verify_data_contract(config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in {1, 2} or not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise RuntimeError("runner supports one GPU or registered two-rank DDP")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError(f"expected exactly {world_size} visible authorized GPU(s)")
    if world_size == 2:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    device = torch.device("cuda", local_rank)
    if config.get("precision", "fp32") != "fp32":
        raise ValueError("FP32 is the registered common numerical contract")

    reference = json.loads(Path(args.safety_reference).read_text())
    if reference.get("kind") != "whole" or reference.get("evaluation_role") != "deployment_validation":
        raise ValueError("safety reference must be full deployment validation")
    if reference.get("manifest_sha256") != config["manifest_sha256"]:
        raise ValueError("safety reference manifest mismatch")
    output = Path(args.output)
    if not args.resume and rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world_size == 2:
        dist.barrier()
    elif not output.is_dir():
        raise ValueError("resume output missing")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    objective = IndependentObjective(CTSystem(config["residual_scale_hu"], config["prior_mode"], config["backbone"]))
    objective = objective.to(device)
    distributed = (DistributedDataParallel(objective, device_ids=[local_rank], output_device=local_rank,
                                            find_unused_parameters=True)
                   if world_size == 2 else objective)
    optimizer = torch.optim.AdamW(distributed.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    cursor = {"epoch": 0, "cursor": 0}
    global_step = 0
    early_stop = EarlyStopState(
        epoch20_gate_enabled=bool(config.get("epoch20_convergence_gate", True))
    )
    best_order = None
    resume = None
    if args.resume:
        resume = read_checkpoint(args.resume)
        if resume["config_sha256"] != fingerprint(config) or resume["variant"] != config["variant"]:
            raise ValueError("resume contract mismatch")
        objective.load_state_dict(resume["objective"])
        optimizer.load_state_dict(resume["optimizer"])
        cursor = resume["stream"]
        global_step = resume["global_step"]
        early_stop = restore_early_stop(resume["early_stop"])
        best_order = tuple(resume["best_order"]) if resume.get("best_order") else None

    train = make_dataset(config, "train", output / f"pixel_access.rank{rank}.jsonl")
    validation = make_dataset(config, "validation", output / f"validation_access.rank{rank}.jsonl")
    stream = BatchStream(train, config["seed"], batch_size=config["micro_batch"], rank=rank,
                         world_size=world_size, **cursor)
    if resume:
        restore_rng(resume["rng"])
    if rank == 0:
        (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
    source_hashes = {str(path.relative_to(Path(__file__).parent)): sha256(path)
                     for path in Path(__file__).parent.rglob("*.py")}
    hashes_path = output / "code_hashes.json"
    if resume and json.loads(hashes_path.read_text()) != source_hashes:
        raise ValueError("resume code changed")
    if rank == 0:
        hashes_path.write_text(json.dumps(source_hashes, indent=2))
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=global_step if resume else None) if rank == 0 else None
    if writer:
        writer.add_text("contract", json.dumps({"config_sha256": fingerprint(config), "variant": config["variant"],
                                                "manifests": config["manifest_sha256"], "max_epochs": 200,
                                                "global_effective_batch": 32}), global_step)
    try:
        while completed_epochs(stream, len(train)) < config["max_epochs"]:
            epoch_start = completed_epochs(stream, len(train)) + 1
            epoch_seconds = 0.0
            epoch_losses = {}
            while completed_epochs(stream, len(train)) < epoch_start:
                started = time.monotonic()
                objective.train()
                optimizer.zero_grad(set_to_none=True)
                losses = {}
                for _ in range(config["accumulation"]):
                    source, target = stream.next()
                    values = distributed(source.to(device), target.to(device))
                    if not torch.isfinite(values["loss"]):
                        raise FloatingPointError("nonfinite loss")
                    (values["loss"] / config["accumulation"]).backward()
                    for key, value in values.items():
                        if key != "prediction":
                            losses[key] = losses.get(key, 0.0) + float(value.detach()) / config["accumulation"]
                norm = torch.sqrt(sum(parameter.grad.float().square().sum()
                                      for parameter in objective.parameters() if parameter.grad is not None))
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                torch.cuda.synchronize(device)
                global_step += 1
                elapsed = time.monotonic() - started
                epoch_seconds += elapsed
                for key, value in losses.items():
                    epoch_losses[key] = epoch_losses.get(key, 0.0) + value
                if rank == 0:
                    row = {"global_step": global_step, "epoch": epoch_start, "variant": config["variant"],
                           "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(norm),
                           "seconds_per_update": elapsed, **losses}
                    _append_jsonl(output / "training.jsonl", row)
                    for key, value in row.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar("train/" + key, value, global_step)

            epoch = completed_epochs(stream, len(train))
            result = None
            decision = {"epoch": epoch, "action": "continue", "reason": "between_validation_checks"}
            if epoch % config["validation_interval_epochs"] == 0:
                if world_size == 2:
                    dist.barrier()
                result = _validate(objective.system, validation, rank, world_size, config)
                if rank == 0:
                    result.update({"epoch": epoch, "variant": config["variant"],
                                   "config_sha256": fingerprint(config),
                                   "manifest_sha256": config["manifest_sha256"]})
                    result["safety_failures"] = safety(result, reference)
                    if result["metrics"]["mae_hu"] > reference["metrics"]["mae_hu"]:
                        result["safety_failures"].append("overall_MAE_above_reference")
                    decision = early_stop.update(epoch, metric_view(result, reference))
                    result["stop_decision"] = decision
                    _append_jsonl(output / "stop_decisions.jsonl", decision)
                    for key, value in result["metrics"].items():
                        writer.add_scalar("val/" + key, value, epoch)
                    for patient, metrics in result["patients"].items():
                        for key, value in metrics.items():
                            writer.add_scalar(f"val/patient/{patient}/{key}", value, epoch)
                if world_size == 2:
                    payload = [decision if rank == 0 else None]
                    dist.broadcast_object_list(payload, src=0)
                    decision = payload[0]
                    dist.barrier()
            if decision["action"] == "reduce_lr":
                for group in optimizer.param_groups:
                    group["lr"] = config["reduced_learning_rate"]
            complete = decision["action"] == "stop" or epoch >= config["max_epochs"]
            if rank == 0:
                state = asdict(early_stop)
                save_checkpoint(output / "last.pt", objective, optimizer, config, epoch, global_step,
                                stream, state, best_order, complete, decision)
                if result is not None:
                    result["last_checkpoint_sha256"] = sha256(output / "last.pt")
                    current_order = selection_key(result)
                    if best_order is None or current_order > best_order:
                        best_order = current_order
                        save_checkpoint(output / "best.pt", objective, optimizer, config, epoch, global_step,
                                        stream, state, best_order, complete, decision)
                        training_sha = sha256(output / "best.pt")
                        result["checkpoint_sha256"] = training_sha
                        (output / "best_validation.json").write_text(json.dumps(result, indent=2))
                        export_deployment(output / "best_deployment.pt", objective.system, config, training_sha)
                        save_checkpoint(output / "last.pt", objective, optimizer, config, epoch, global_step,
                                        stream, state, best_order, complete, decision)
                    (output / f"validation_epoch_{epoch:03d}.json").write_text(json.dumps(result, indent=2))
                eta = {"epoch": epoch, "global_step": global_step,
                       "remaining_training_seconds_estimate": remaining_epoch_updates(stream, len(train), 200)
                       * (epoch_seconds / max(1, optimizer_updates_per_epoch(len(train)))),
                       "validation_time_excluded": True, "status": "TRAINING_ONLY_ESTIMATE"}
                (output / "throughput_eta.json").write_text(json.dumps(eta, indent=2))
                writer.flush()
            if complete:
                break
    finally:
        if writer:
            writer.close()
        if world_size == 2:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
