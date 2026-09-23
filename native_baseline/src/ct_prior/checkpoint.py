import os
import random
import numpy as np
import torch
from .config import fingerprint

TRAINING_SCHEMA = "ct_prior_independent_training_v3"
DEPLOYMENT_SCHEMA = "ct_prior_independent_deployment_v2"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, objective, optimizer, config, epoch, global_step, stream,
                    early_stop, best_order, training_complete=False, stop_decision=None):
    payload = {"schema": TRAINING_SCHEMA, "config": config, "config_sha256": fingerprint(config),
               "variant": config["variant"], "epoch": int(epoch), "global_step": int(global_step),
               "objective": objective.state_dict(), "optimizer": optimizer.state_dict(),
               "stream": stream.state_dict(), "rng": rng_state(), "early_stop": early_stop,
               "best_order": best_order, "training_complete": bool(training_complete),
               "stop_decision": stop_decision}
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != TRAINING_SCHEMA or fingerprint(checkpoint["config"]) != checkpoint["config_sha256"]:
        raise ValueError("invalid independent training checkpoint")
    if {"stage", "parent_sha256", "s0_sha256", "stage_complete"}.intersection(checkpoint):
        raise ValueError("staged checkpoint cannot resume an independent task")
    return checkpoint


def export_deployment(path, system, config, training_sha):
    torch.save({"schema": DEPLOYMENT_SCHEMA, "system": system.state_dict(), "config": config,
                "variant": config["variant"], "training_sha256": training_sha,
                "teacher_included": False}, path)
