from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Mapping
import torch

def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def state_sha256(state: Mapping[str, torch.Tensor], common_only: bool = False) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if common_only and any(token in "." + key for token in (".carrier.", ".a29.", ".b29.")):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
