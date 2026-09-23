"""Run the standalone reference or an explicitly integrated native bridge.
No remote execution, no automatic patient training, no hidden historical-B0 substitution.
"""
import argparse
from dataclasses import asdict
import importlib
import json
import random
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from .config import ModelConfig,config_hash
from .model import CESC
from .training import train_epoch
from .evaluate import evaluate,evaluate_statistics
from .checkpoint import save_checkpoint,load_checkpoint
from .audit import append_jsonl,write_json,file_sha256,module_sha256,implementation_sha256

def import_bridge(spec,config):
    module,name=spec.split(':',1)
    return getattr(importlib.import_module(module),name)(config)

def native_gate(contract,config,authorized,preflight):
    if contract.get('source')=='SYNTHETIC_ONLY':return
    if not authorized:raise RuntimeError('Patient training requires --authorize-patient-training.')
    if not preflight:raise RuntimeError('Patient training requires a verified --preflight-report.')
    report=json.loads(Path(preflight).read_text())
    if report.get('passed') is not True or report.get('bridge_contract')!=contract or report.get('implementation_sha256')!=implementation_sha256():
        raise RuntimeError('Preflight is missing, failed, or belongs to another native contract.')
    for key in ('fixed_b0_sha256','manifest_sha256'):
        value=contract.get(key,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError(f'Native contract requires actual SHA256: {key}.')
    if contract.get('patient_split')!='train6_val3_lockedtest1':
        raise RuntimeError('Native main comparison must preserve train6/val3/lockedtest1.')
    if contract.get('main_series')!='3mm_B30':raise RuntimeError('Native main series must be 3mm_B30.')
    if not contract.get('native_reconstruction_loss_verified',False):
        raise RuntimeError('Audit native L1+0.1*MSE standardized-HU loss before using this runner.')
    effective=ModelConfig(**config['model'])
    if contract.get('hu_center')!=effective.hu_center or contract.get('hu_scale')!=effective.hu_scale:
        raise RuntimeError('Native HU normalization does not match configured center/scale.')
    if config['model']['s_ct']!=contract.get('s_ct'):
        raise RuntimeError('Exact native s_ct mismatch; do not silently use rounded demo scale.')

def run(config,bridge_spec,outdir,device='cpu',stats_checkpoint=None,resume=None,
        authorize=False,preflight=None):
    torch.set_num_threads(int(config.get('cpu_threads',2)))
    seed=int(config.get('seed',123456))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    bridge=import_bridge(bridge_spec,config);contract=bridge.contract()
    native_gate(contract,config,authorize,preflight)
    cfg=ModelConfig(**config['model']);variant=config.get('variant','full');stage=config['stage']
    model=CESC(cfg,variant).to(device).set_stage(stage)
    provenance={'bridge_contract':contract,'config_sha256':config_hash(config),
                'stage':stage,'variant':variant,'code_version':'0.31.1-rebuilt',
                'implementation_sha256':implementation_sha256(),'model_config':asdict(cfg),
                'stats_checkpoint_sha256':file_sha256(stats_checkpoint) if stats_checkpoint else None}
    if stage=='correction' and variant in ('full','diagonal'):
        if not stats_checkpoint:raise RuntimeError('Full/Diagonal require the same selected statistics checkpoint.')
        data=torch.load(stats_checkpoint,map_location='cpu',weights_only=False)
        source=data.get('provenance',{})
        if source.get('stage')!='statistics' or source.get('bridge_contract')!=contract:
            raise RuntimeError('Statistics checkpoint is not matched to the fixed B0/data contract.')
        if source.get('implementation_sha256')!=provenance['implementation_sha256'] or source.get('model_config')!=asdict(cfg):
            raise RuntimeError('Statistics code/model configuration mismatch.')
        state={k.removeprefix('statistics.'):v for k,v in data['model'].items() if k.startswith('statistics.')}
        model.statistics.load_state_dict(state,strict=True)
    out=Path(outdir)
    if resume:
        if not out.is_dir():raise RuntimeError('Resume output directory must already exist.')
    else:out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config);write_json(out/'provenance.json',provenance)
    write_json(out/'parameters.json',model.parameter_report())
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters:raise ValueError('No trainable parameters; B0 is evaluated, not trained, by this runner.')
    optimizer=torch.optim.AdamW(parameters,lr=float(config['lr']),weight_decay=float(config.get('weight_decay',0.0)))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(config['epochs']),eta_min=float(config['min_lr']))
    start=1;best=float('inf') if stage=='statistics' else -float('inf')
    if resume:
        data=load_checkpoint(resume,model,optimizer,scheduler,provenance)
        start=data['epoch']+1;best=data['extra']['best']
    writer=None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(out/'tensorboard'))
    except ImportError:
        if contract['source']!='SYNTHETIC_ONLY':raise RuntimeError('Install tensorboard for native evidence logging.')
        append_jsonl(out/'events.jsonl',{'kind':'environment','tensorboard':'UNAVAILABLE','source':contract['source']})
    frozen_hash=module_sha256(model.statistics) if stage=='correction' and model.statistics else None
    bridge.assert_frozen_b0()
    # Fixed B0 result is collected in the same validation pass, not a historical epoch curve.
    try:
        for epoch in range(start,int(config['epochs'])+1):
            lr=optimizer.param_groups[0]['lr']
            train=train_epoch(model,bridge,optimizer,epoch,int(config['accumulation']),device,int(config['micro_batch']))
            scheduler.step()
            event={'kind':'train','source':contract['source'],'stage':stage,'variant':variant,
                   'epoch':epoch,'lr_used':lr,'lr_next':optimizer.param_groups[0]['lr'],**train}
            append_jsonl(out/'events.jsonl',event)
            if writer:
                writer.add_scalar('train/loss',train['loss'],epoch);writer.add_scalar('train/lr_used',lr,epoch)
            should_eval=epoch%int(config['val_interval'])==0 or epoch==int(config['epochs'])
            improved=False
            if should_eval:
                result=evaluate_statistics(model,bridge,device) if stage=='statistics' else evaluate(model,bridge,device)
                event={'kind':'validation_main','source':contract['source'],'stage':stage,
                       'variant':variant,'epoch':epoch,'result':result}
                append_jsonl(out/'events.jsonl',event)
                score=result['patient_equal_mean']['working_nll'] if stage=='statistics' else result['candidate']['patient_equal_mean']['psnr']
                improved=score<best if stage=='statistics' else score>best
                if improved:best=score
                if writer:
                    writer.add_scalar('validation/selection_score',score,epoch)
                    if stage=='statistics':
                        for k,v in result['patient_equal_mean'].items():writer.add_scalar('validation/'+k,v,epoch)
                    else:
                        for group in ('candidate','fixed_b0'):
                            for k,v in result[group]['patient_equal_mean'].items():writer.add_scalar('validation/'+group+'/'+k,v,epoch)
                        for k,v in result['delta'].items():writer.add_scalar('validation/delta/'+k,v,epoch)
            if stage=='correction' and epoch in config.get('probe_epochs',[40,90,130]):
                for alpha in (0.0,0.5,1.0):
                    result=evaluate(model,bridge,device,alpha)
                    append_jsonl(out/'events.jsonl',{'kind':'probe_eval_only','source':contract['source'],
                         'epoch':epoch,'alpha':alpha,'result':result,'eligible_for_best':False})
                    if writer:
                        writer.add_scalar(f'probe_alpha_{alpha}/psnr',result['candidate']['patient_equal_mean']['psnr'],epoch)
            if frozen_hash and module_sha256(model.statistics)!=frozen_hash:
                raise RuntimeError('Frozen statistics weights changed during correction training.')
            extra={'best':best,'checkpoint_boundary':'end_of_epoch','mid_epoch_resume_supported':False}
            save_checkpoint(out/'last.pt',model,optimizer,scheduler,epoch,provenance,extra)
            if improved:save_checkpoint(out/'best.pt',model,optimizer,scheduler,epoch,provenance,extra)
            print(json.dumps(event,ensure_ascii=False),flush=True)
    finally:
        if writer:writer.close()
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--bridge',default='bridges.native_template:NativeBridge')
    p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--stats-checkpoint');p.add_argument('--resume')
    p.add_argument('--authorize-patient-training',action='store_true');p.add_argument('--preflight-report')
    args=p.parse_args()
    run(json.loads(Path(args.config).read_text()),args.bridge,args.out,args.device,args.stats_checkpoint,args.resume,
        args.authorize_patient_training,args.preflight_report)
if __name__=='__main__':main()
