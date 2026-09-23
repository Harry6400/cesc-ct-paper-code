import copy
import json
from pathlib import Path
import pytest
import torch
from cesc.metrics import PatientMetrics,paired_delta
from cesc.checkpoint import save_checkpoint,load_checkpoint
from cesc.model import CESC
from cesc.audit import optimizer_coverage,module_sha256
from cesc.losses import reconstruction_loss
from cesc.runner import run,native_gate
from cesc.bridge import Batch
from bridges.native_template import NativeBridge
from scripts.run_ddp import merge_validation_results


def test_patient_equal_not_slice_equal():
    m=PatientMetrics();m.add('a','1',{'psnr':10});m.add('a','2',{'psnr':20});m.add('b','1',{'psnr':30})
    assert m.summary()['patient_equal_mean']['psnr']==22.5

def test_duplicate_key_rejected():
    m=PatientMetrics();m.add('a','1',{'psnr':1})
    with pytest.raises(ValueError):m.add('a','1',{'psnr':2})

def test_unpaired_rows_rejected():
    a=PatientMetrics();b=PatientMetrics();a.add('a','1',{'psnr':1});b.add('a','2',{'psnr':1})
    with pytest.raises(ValueError):paired_delta(a,b)

def test_distributed_validation_merge_matches_single_pass():
    rows=[]
    candidate=PatientMetrics();baseline=PatientMetrics()
    for index in range(6):
        patient=f'p{index % 2}';slice_id=str(index)
        current={'psnr_db':40.0+index,'ssim':0.9+index/1000,'mae_hu':12.0-index}
        fixed={'psnr_db':39.0+index,'ssim':0.89+index/1000,'mae_hu':13.0-index}
        candidate.add(patient,slice_id,current);baseline.add(patient,slice_id,fixed)
        rows.append({'patient_id':patient,'slice_id':slice_id,
                     'candidate':current,'fixed_b0':fixed})
    rank_results=[{'slice_metrics':rows[0::2]},{'slice_metrics':rows[1::2]}]
    assert merge_validation_results(rank_results,6)==paired_delta(candidate,baseline)

def test_distributed_validation_merge_rejects_missing_slice():
    row={'patient_id':'p','slice_id':'0','candidate':{'psnr_db':1.0},
         'fixed_b0':{'psnr_db':0.0}}
    with pytest.raises(RuntimeError):merge_validation_results([{'slice_metrics':[row]},
                                                               {'slice_metrics':[]}],2)

def test_empty_metrics_rejected():
    with pytest.raises(ValueError):PatientMetrics().summary()

@pytest.mark.parametrize('variant',['plain','full','diagonal'])
def test_checkpoint_next_update_exact(cfg,tmp_path,variant):
    a=CESC(cfg,variant).set_stage('correction')
    opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
    def update(m,o,s):
        x=torch.randn(2,1,8,8)+1000;b=x+.3;t=x+.7
        o.zero_grad();loss=reconstruction_loss(m(x,b),t,cfg.s_ct);loss.backward();o.step();s.step()
        return loss.detach()
    update(a,opt,sch);save_checkpoint(tmp_path/'c.pt',a,opt,sch,1,{'x':'test'})
    reference=update(a,opt,sch);state={k:v.clone() for k,v in a.state_dict().items()}
    b=CESC(cfg,variant).set_stage('correction');o=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=.001)
    s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=10)
    load_checkpoint(tmp_path/'c.pt',b,o,s,{'x':'test'})
    actual=update(b,o,s);torch.testing.assert_close(reference,actual,rtol=0,atol=0)
    for k,v in state.items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    assert opt.param_groups[0]['lr']==o.param_groups[0]['lr']

def test_checkpoint_rejects_provenance(cfg,tmp_path):
    m=CESC(cfg);save_checkpoint(tmp_path/'c.pt',m,None,None,0,{'data':'a'})
    with pytest.raises(RuntimeError):load_checkpoint(tmp_path/'c.pt',m,expected_provenance={'data':'b'})

def test_optimizer_coverage(cfg):
    m=CESC(cfg).set_stage('correction');o=torch.optim.Adam([p for p in m.parameters() if p.requires_grad])
    optimizer_coverage(m,o)
    bad=torch.optim.Adam(list(m.correction.parameters())[:1])
    with pytest.raises(RuntimeError):optimizer_coverage(m,bad)

def test_native_stub_fails_honestly():
    with pytest.raises(NotImplementedError):NativeBridge({})

def test_native_gate_rejects_unauthorized():
    with pytest.raises(RuntimeError):native_gate({'source':'NATIVE'}, {},False,None)

def test_batch_type_guard():
    x=torch.zeros(1,1,8,8,dtype=torch.float64)
    with pytest.raises(ValueError):Batch(x,x,x,['p'],['s']).validate()

def test_end_to_end_synthetic_stages(tmp_path):
    cfg=json.loads(Path('configs/demo.json').read_text());cfg['epochs']=1;cfg['probe_epochs']=[1]
    stats=copy.deepcopy(cfg);stats['stage']='statistics'
    root=run(stats,'bridges.synthetic:SyntheticBridge',tmp_path/'stats')
    for variant in ('full','diagonal','plain'):
        c=copy.deepcopy(cfg);c['variant']=variant
        out=run(c,'bridges.synthetic:SyntheticBridge',tmp_path/variant,
                stats_checkpoint=str(root/'best.pt') if variant!='plain' else None)
        rows=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        assert any(x['kind']=='validation_main' for x in rows)
        assert sum(x['kind']=='probe_eval_only' for x in rows)==3
        assert all(x.get('eligible_for_best') is False for x in rows if x['kind']=='probe_eval_only')
        assert (out/'best.pt').is_file() and (out/'last.pt').is_file()
