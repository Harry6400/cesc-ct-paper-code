"""Opt-in runner guards. The caller still owns the native dataset/optimizer/scheduler.
Nothing in this module launches jobs or changes a learning rate.
"""
from __future__ import annotations
import copy, json, math, random
from pathlib import Path
import numpy as np
import torch
from .common import json_digest
from .audit import assert_resume_compatible


def assert_optimizer_coverage(model, optimizer):
    wanted={id(p):name for name,p in model.named_parameters() if p.requires_grad}
    seen={}
    for gi,group in enumerate(optimizer.param_groups):
        for p in group['params']:
            key=id(p)
            if key not in wanted: raise ValueError(f'Foreign/frozen optimizer parameter in group {gi}')
            if key in seen: raise ValueError(f'Duplicate optimizer parameter: {wanted[key]}')
            seen[key]=gi
    if wanted.keys()!=seen.keys():
        raise ValueError('Missing trainable optimizer parameters: '+str([wanted[k] for k in wanted.keys()-seen.keys()]))
    return {'trainable_parameters':len(wanted), 'groups':len(optimizer.param_groups),
            'covered_numel':sum(p.numel() for p in model.parameters() if p.requires_grad)}


class LRTraceGuard:
    """Compare the actual LR USED at each optimizer update (one-based step).
    group_reference_indices maps each candidate group to one historical group.
    prepare() goes immediately before optimizer.step(), commit() immediately after.
    Call neither on intermediate accumulation microbatches. Never call if an AMP
    update was skipped (FP32 formal default). Scheduler order stays native.
    """
    def __init__(self, rows, group_reference_indices, completed_steps=0):
        if not rows or not group_reference_indices: raise ValueError('Trace/mapping cannot be empty')
        self.rows=copy.deepcopy(rows); self.mapping=tuple(group_reference_indices)
        for i,row in enumerate(self.rows,1):
            if row.get('optimizer_step')!=i: raise ValueError('Historical trace must be contiguous, one-based')
            if not row.get('lrs_used') or not all(math.isfinite(float(x)) and x>=0 for x in row['lrs_used']):
                raise ValueError('Nonfinite/negative/missing historical LR')
            if any(type(j) is not int or j<0 or j>=len(row['lrs_used']) for j in self.mapping):
                raise ValueError('Invalid reference group mapping')
        if type(completed_steps) is not int or not 0<=completed_steps<=len(rows): raise ValueError('Bad cursor')
        self.completed_steps=completed_steps;self.pending=None
        self.trace_sha256=json_digest(self.rows);self.mapping_sha256=json_digest(self.mapping)

    def prepare(self, optimizer, optimizer_step):
        if self.pending is not None: raise RuntimeError('Previous optimizer update not committed')
        if optimizer_step!=self.completed_steps+1: raise ValueError('Repeated/skipped optimizer step')
        if optimizer_step>len(self.rows): raise ValueError('Historical LR trace exhausted; do not extrapolate')
        actual=[float(g['lr']) for g in optimizer.param_groups]
        expected=[float(self.rows[optimizer_step-1]['lrs_used'][j]) for j in self.mapping]
        if len(actual)!=len(expected) or any(not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-14) for a,b in zip(actual,expected)):
            raise ValueError(f'LR mismatch at update {optimizer_step}: actual={actual}, reference={expected}')
        self.pending={'optimizer_step':optimizer_step,'lrs_used':actual,
                      'trace_sha256':self.trace_sha256,'mapping_sha256':self.mapping_sha256}
        return copy.deepcopy(self.pending)

    def commit(self):
        if self.pending is None: raise RuntimeError('prepare() required before commit()')
        record=self.pending;self.completed_steps+=1;self.pending=None
        return record

    def state_dict(self):
        if self.pending is not None: raise RuntimeError('Checkpoint only at committed optimizer boundaries')
        return {'completed_steps':self.completed_steps,'trace_sha256':self.trace_sha256,
                'mapping_sha256':self.mapping_sha256}

    def load_state_dict(self,state):
        if state['trace_sha256']!=self.trace_sha256 or state['mapping_sha256']!=self.mapping_sha256:
            raise ValueError('LR trace or group mapping changed on resume')
        k=state['completed_steps']
        if type(k) is not int or not 0<=k<=len(self.rows): raise ValueError('Bad resume LR cursor')
        self.completed_steps=k;self.pending=None


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        if not torch.cuda.is_available(): raise ValueError('Cannot restore CUDA RNG on CPU-only runtime')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def checkpoint_payload(model,optimizer,scheduler,identity,lr_guard,stream_state,epoch):
    """Caller MUST serialize sampler/augmentation/order states in stream_state.
    This cannot capture arbitrary DataLoader worker queues. Use the native runner's
    verified exact-resume protocol; checkpoint only at optimizer boundaries.
    """
    assert_resume_compatible(identity,identity)
    assert_optimizer_coverage(model,optimizer)
    if stream_state is None: raise ValueError('Explicit native stream state required')
    return copy.deepcopy({'identity':identity,'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'lr_guard':lr_guard.state_dict(),'stream_state':stream_state,'rng':capture_rng(),'epoch':int(epoch)})


def restore_checkpoint(payload,model,optimizer,scheduler,identity,lr_guard):
    assert_resume_compatible(payload['identity'],identity)
    model.load_state_dict(payload['model'],strict=True)
    # Scheduler must already have been constructed before optimizer restore.
    optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    lr_guard.load_state_dict(payload['lr_guard']);assert_optimizer_coverage(model,optimizer)
    restore_rng(payload['rng'])
    return copy.deepcopy(payload['stream_state']),payload['epoch']


def append_jsonl(path, record):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
