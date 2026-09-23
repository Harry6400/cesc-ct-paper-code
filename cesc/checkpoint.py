"""Epoch-boundary checkpointing. Load only files you trust: torch pickle is executable."""
import os
import random
from pathlib import Path
import numpy as np
import torch

def capture_rng()->dict:
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state:dict):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():raise RuntimeError('CUDA RNG checkpoint cannot resume on CPU.')
        torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(path,model,optimizer,scheduler,epoch:int,provenance:dict,extra:dict|None=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save({'format':'cesc-rebuilt-epoch-v1','model':model.state_dict(),
        'optimizer':optimizer.state_dict() if optimizer else None,
        'scheduler':scheduler.state_dict() if scheduler else None,
        'epoch':epoch,'rng':capture_rng(),'provenance':provenance,'extra':extra or {}},tmp)
    os.replace(tmp,path)

def load_checkpoint(path,model,optimizer=None,scheduler=None,expected_provenance:dict|None=None,restore:bool=True):
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('format')!='cesc-rebuilt-epoch-v1':raise ValueError('Not a CESC rebuild checkpoint.')
    if expected_provenance is not None and data['provenance']!=expected_provenance:
        raise RuntimeError('Checkpoint provenance/config does not match; refusing silent resume.')
    model.load_state_dict(data['model'],strict=True)
    if optimizer is not None:optimizer.load_state_dict(data['optimizer'])
    if scheduler is not None:scheduler.load_state_dict(data['scheduler'])
    if restore:restore_rng(data['rng'])
    return data
