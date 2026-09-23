"""Local integration audit: performs a disposable update, never launches a patient training run."""
import argparse,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cesc.config import ModelConfig
from cesc.model import CESC
from cesc.runner import import_bridge
from cesc.losses import reconstruction_loss
from cesc.checkpoint import save_checkpoint,load_checkpoint
from cesc.audit import module_sha256,optimizer_coverage,write_json,implementation_sha256
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--bridge',required=True)
p.add_argument('--device',default='cpu');p.add_argument('--out',required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());torch.set_num_threads(2)
bridge=import_bridge(a.bridge,cfg);contract=bridge.contract()
report={'passed':False,'source':contract.get('source'),'device':a.device,'bridge_contract':contract,
        'implementation_sha256':implementation_sha256(),'checks':{},'not_a_training_authorization':True}
try:
    bridge.assert_frozen_b0();batch=next(iter(bridge.train_batches(1))).validate(cfg['micro_batch']).to(a.device)
    m=CESC(ModelConfig(**cfg['model']),'full').to(a.device).set_stage('correction')
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['lr'])
    optimizer_coverage(m,opt);before=module_sha256(m.statistics)
    with torch.no_grad():pred=m(batch.center_hu,batch.b0_hu)
    torch.testing.assert_close(pred,batch.b0_hu,rtol=0,atol=0)
    report['checks']['initial_parity_max_hu']=float((pred-batch.b0_hu).abs().max())
    def update(model,optimizer):
        optimizer.zero_grad();out=model(batch.center_hu,batch.b0_hu)
        loss=reconstruction_loss(out,batch.target_hu,model.cfg.s_ct);loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite preflight loss.')
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient.')
        optimizer.step();return float(loss.detach())
    update(m,opt)
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'ckpt.pt';prov={'preflight':True,'contract':contract}
        save_checkpoint(path,m,opt,None,1,prov)
        expected=update(m,opt);expected_state=module_sha256(m)
        load_checkpoint(path,m,opt,expected_provenance=prov)
        actual=update(m,opt)
        if expected!=actual or module_sha256(m)!=expected_state:raise RuntimeError('Next-update resume mismatch.')
    if module_sha256(m.statistics)!=before:raise RuntimeError('Frozen statistics changed.')
    bridge.assert_frozen_b0()
    report['checks'].update(optimizer_coverage=True,finite_backward=True,statistics_frozen=True,
         checkpoint_restore_next_update_exact=True,native_b0_audit=True)
    report['passed']=True
except Exception as e:
    report['error']=repr(e);write_json(a.out,report);raise
write_json(a.out,report);print(json.dumps(report,indent=2))
