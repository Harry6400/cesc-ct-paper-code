from pathlib import Path
import hashlib
import json
import torch

def file_sha256(path:str|Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def module_sha256(module:torch.nn.Module)->str:
    h=hashlib.sha256()
    for name,t in sorted(module.state_dict().items()):
        a=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def append_jsonl(path:str|Path,event:dict):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')

def write_json(path:str|Path,data:dict):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def optimizer_coverage(model,optimizer):
    expected={id(p) for p in model.parameters() if p.requires_grad}
    actual=[id(p) for g in optimizer.param_groups for p in g['params']]
    if set(actual)!=expected or len(actual)!=len(set(actual)):
        raise RuntimeError('Optimizer parameters do not exactly match the trainable set.')

def implementation_sha256()->str:
    h=hashlib.sha256()
    root=Path(__file__).parent
    for path in sorted(root.glob('*.py')):
        h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()
