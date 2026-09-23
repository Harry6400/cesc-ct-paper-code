"""Backend-neutral scalar logging; the native runner owns TensorBoard/JSONL."""
from __future__ import annotations
import math
from .runtime import append_jsonl

def scalar_tree(value,prefix=''):
    if isinstance(value,dict):
        for key,item in value.items():yield from scalar_tree(item,f'{prefix}/{key}' if prefix else str(key))
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value):yield from scalar_tree(item,f'{prefix}/{i}')
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        if not math.isfinite(float(value)):raise ValueError('Nonfinite logged scalar: '+prefix)
        yield prefix,float(value)

def log_probe(writer,jsonl_path,epoch,record):
    if record.get('eligible_for_checkpoint_selection') is not False:
        raise ValueError('Probe record must be excluded from checkpoint selection')
    # A writer may be None for JSONL-only smoke tests, never for the formal run.
    prefix='probe/'+record['label']
    if writer is not None:
        for name,value in scalar_tree(record,prefix):writer.add_scalar(name,value,int(epoch))
    append_jsonl(jsonl_path,dict(record,epoch=int(epoch),record_type='eval_probe'))
