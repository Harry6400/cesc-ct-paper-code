"""Five-slice views on the existing, hash-bound development ledgers."""
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from .shared.mayo_data import MayoTrainingDataset, MayoValidationDataset
from .shared.data import LidcBaselineDataset
from .restoration_backbone import hu_to_model


def seed_from(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def _pack(qd, fd, patient, series, center, top=0, left=0):
    if "L506" in str(patient) or "L506" in str(series):
        raise RuntimeError("locked patient forbidden")
    y = torch.from_numpy(np.ascontiguousarray(qd, dtype=np.float32))
    target = torch.from_numpy(np.ascontiguousarray(fd, dtype=np.float32)).clamp(-1024,3072)
    return {"y":hu_to_model(y), "target_hu":target, "patient_id":patient,
            "series_id":series, "slice_id":int(center), "top":top, "left":left}


class MayoFive(Dataset):
    def __init__(self, c, role, audit):
        self.role = role
        self.context_mode = c.get("context_mode", "real_five")
        if self.context_mode not in {"real_five", "center_repeat"}:
            raise ValueError("context_mode must be real_five or center_repeat")
        self.base = (MayoTrainingDataset(c["paths"]["manifest"], "redcnn", audit, seed=c["seed"])
                     if role == "train" else MayoValidationDataset(c["paths"]["manifest"], "redcnn", audit))
        if role not in {"train", "validation"}:
            raise ValueError("development roles only")

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        if self.role == "train":
            self.base.set_epoch(epoch)

    def __getitem__(self, index):
        record_id, center = self.base.samples[index]
        r = self.base.records[record_id]
        qd = self.base.volumes.load(r, "qd_path", self.role)
        fd = self.base.volumes.load(r, "fd_path", self.role)
        z = [min(max(center+i, 0), len(qd)-1) for i in (-2,-1,0,1,2)]
        if self.role == "train":
            top, left = self.base._coordinate(r, center)
            q = np.stack([qd[i,top:top+128,left:left+128] for i in z])
            f = fd[center: center+1,top:top+128,left:left+128]
        else:
            top=left=0
            q, f = np.stack([qd[i] for i in z]), fd[center:center+1]
        if self.context_mode == "center_repeat":
            q = np.repeat(q[2:3], 5, axis=0)
        return _pack(q, f, r["patient_id"], r["patient_id"], center, top, left)


class LidcFive(LidcBaselineDataset):
    def __init__(self, c, role, audit):
        super().__init__(c["paths"]["manifest"], c["paths"]["split_manifest"], role,
                         "redcnn", audit, seed=c["seed"])

    def __getitem__(self, index):
        pair_id, sample = divmod(index, self.samples_per_series)
        p = self.pairs[pair_id]
        z, top, left = self._coordinate(p, sample)
        q = self._load(p["low_path"], p, "low")
        f = self._load(p["full_path"], p, "full")
        return _pack(q[z:z+5,top:top+128,left:left+128],
                     f[z+2:z+3,top:top+128,left:left+128],p["patient_id"],p["series_id"],z+2,top,left)


def make_dataset(c, role, audit):
    if role not in {"train", "validation"}:
        raise ValueError("development roles only")
    return (MayoFive if c["dataset"] == "mayo" else LidcFive)(c, role, audit)


class BatchStream:
    """Deterministic ledger cursor; no hidden prefetch/RNG state on resume."""
    def __init__(self, dataset, seed, batch_size=8, epoch=0, cursor=0, rank=0, world_size=1):
        if len(dataset) < 32:
            raise ValueError("training ledger too small")
        if world_size < 1 or not 0 <= rank < world_size or batch_size % world_size:
            raise ValueError("invalid distributed stream partition")
        self.dataset,self.seed,self.batch_size = dataset,seed,batch_size
        self.epoch,self.cursor = epoch,cursor
        self.rank,self.world_size = rank,world_size

    def next(self):
        n = len(self.dataset)//32*32
        if self.cursor >= n:
            self.epoch += 1
            self.cursor = 0
        self.dataset.set_epoch(self.epoch)
        order = np.random.default_rng(self.seed+self.epoch).permutation(len(self.dataset))[:n]
        ids = order[self.cursor:self.cursor+self.batch_size]
        self.cursor += self.batch_size
        ids = ids[self.rank::self.world_size]
        rows = [self.dataset[int(i)] for i in ids]
        return torch.stack([r["y"] for r in rows]),torch.stack([r["target_hu"] for r in rows])

    def state_dict(self):
        return {"epoch":self.epoch,"cursor":self.cursor}
