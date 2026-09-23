import contextlib
import math
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .shared.metrics import patch_metrics
from .data import seed_from


def region_metrics(pred, target):
    # Exact v10s/v10t region definitions, not the external campaign's 240/50 thresholds.
    target = target.float()
    error = pred.float()-target
    body, dense = target >= -900, target >= 200
    gx = F.pad(target[...,1:]-target[...,:-1],(0,1,0,0))
    gy = F.pad(target[...,1:,:]-target[...,:-1,:],(0,0,0,1))
    gradient = (gx.square()+gy.square()).sqrt()
    threshold = torch.quantile(gradient[body],0.9) if body.any() else float("inf")
    mask = body & (gradient >= threshold)
    def mean(m, absolute=True):
        values = error[m]
        return float((values.abs() if absolute else values).mean()) if values.numel() else math.nan
    return {"high_density_mae_hu":mean(dense),"high_gradient_mae_hu":mean(mask),
            "high_density_bias_hu":mean(dense,False)}


def summarize(rows):
    if not rows:
        raise ValueError("empty evaluation")
    keys = [k for k in rows[0] if k not in {"patient_id","series_id","slice_id"}]
    def means(values):
        return {k:float(np.mean([v[k] for v in values if math.isfinite(v[k])]))
                if any(math.isfinite(v[k]) for v in values) else math.nan for k in keys}
    series = defaultdict(list)
    for r in rows:
        series[(r["patient_id"],r["series_id"])].append(r)
    patients = defaultdict(list)
    for (p,_),values in series.items():
        patients[p].append(means(values))
    by_patient = {p:means(v) for p,v in patients.items()}
    return {"aggregation":"slice_or_patch_to_series_to_patient_equal",
            "metrics":means(list(by_patient.values())),"patients":by_patient,"samples":len(rows)}


def starts(length, patch=128, overlap=32):
    if length < patch:
        raise ValueError("image smaller than patch")
    values = list(range(0,length-patch+1,patch-overlap))
    return values if values[-1] == length-patch else values+[length-patch]


@torch.no_grad()
def infer_image(model, y, key, teacher=None, target=None, use_bf16=False, patch=128,
                patch_streams=1):
    if not isinstance(patch_streams,int) or patch_streams < 1:
        raise ValueError("patch_streams must be a positive integer")
    device = next(model.parameters()).device
    _,h,w = y.shape
    out = torch.zeros((1,h,w),device=device)
    total = torch.zeros_like(out)
    axis = torch.hann_window(patch,periodic=False,device=device).clamp_min(1e-3)
    window = (axis[:,None]*axis[None]).clamp_min(1e-6)
    y_device=y.to(device);target_device=target.to(device) if target is not None else None
    positions=[(top,left) for top in starts(h,patch) for left in starts(w,patch)]
    streams=([torch.cuda.Stream(device=device) for _ in range(patch_streams)]
             if device.type=="cuda" and patch_streams>1 else None)
    for offset in range(0,len(positions),patch_streams):
        group=positions[offset:offset+patch_streams];predictions=[]
        for index,(top,left) in enumerate(group):
            stream=streams[index] if streams else contextlib.nullcontext()
            context=torch.cuda.stream(stream) if streams else stream
            with context:
                crop=y_device[:,top:top+patch,left:left+patch][None]
                generator=torch.Generator(device=device).manual_seed(seed_from(8123456,key,top,left))
                noise=torch.randn((1,32,patch//8,patch//8),generator=generator,device=device)
                amp=(torch.autocast(device_type=device.type,dtype=torch.bfloat16)
                     if use_bf16 else contextlib.nullcontext())
                with amp:
                    if teacher is None:
                        prediction=model(crop,noise)
                    else:
                        if target_device is None:
                            raise ValueError("oracle diagnostic needs labeled target")
                        target_crop=target_device[:,top:top+patch,left:left+patch][None]
                        prior=teacher.teacher_prior(crop,target_crop)
                        prediction=model.restorer.reconstruct(crop,model.read_prior(prior),model.scale_hu)
                predictions.append(prediction)
        if streams:
            current=torch.cuda.current_stream(device)
            for stream in streams[:len(group)]: current.wait_stream(stream)
        for prediction,(top,left) in zip(predictions,group):
            out[:,top:top+patch,left:left+patch] += prediction[0].float()*window
            total[:,top:top+patch,left:left+patch] += window
    return (out/total).clamp(-1024,3072).cpu()


@torch.no_grad()
def evaluate(model, dataset, kind="whole", teacher=None, use_bf16=False, indices=None,
             patch_streams=1):
    was_training = model.training
    model.eval()
    rows=[]
    if indices is None:
        indices = (range(len(dataset)) if kind == "whole" else
                   np.linspace(0,len(dataset)-1,min(96,len(dataset))).round().astype(int))
    try:
        for i in indices:
            r=dataset[int(i)];y=r["y"];target=r["target_hu"]
            key=(r["patient_id"],r["series_id"],r["slice_id"],r["top"],r["left"])
            if kind == "patch":
                rng=np.random.default_rng(seed_from(9123456,key))
                top=int(rng.integers(0,y.shape[-2]-127));left=int(rng.integers(0,y.shape[-1]-127))
                y=y[:,top:top+128,left:left+128];target=target[:,top:top+128,left:left+128]
                key=(*key,top,left)
            prediction=infer_image(model,y,key,teacher,target,use_bf16,patch_streams=patch_streams)
            metrics=patch_metrics(((prediction+1024)/4096)[None],((target+1024)/4096)[None])
            metrics["psnr_db"] = metrics.pop("psnr_hu")
            metrics["ssim"] = metrics.pop("ssim_hu")
            metrics.update(region_metrics(prediction,target))
            rows.append({"patient_id":r["patient_id"],"series_id":r["series_id"],"slice_id":r["slice_id"],**metrics})
    finally:
        model.train(was_training)
    result=summarize(rows)
    result.update({"kind":kind,"evaluation_role":"teacher_oracle_diagnostic" if teacher else "deployment_validation",
                   "regions":"target>=200HU; top10percent gradients inside target>=-900HU", "rows":rows})
    return result


def main():
    import argparse
    import json
    from pathlib import Path
    from .config import load_config,verify_data_contract,sha256
    from .system import CTSystem
    from .data import make_dataset
    parser=argparse.ArgumentParser(description='Validation-only deployment checkpoint evaluation')
    parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--gpu-authorized',action='store_true')
    args=parser.parse_args()
    if not args.gpu_authorized: parser.error('separate GPU evaluation authorization required')
    c=load_config(args.config);verify_data_contract(c)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!='ct_prior_independent_deployment_v2' or checkpoint.get('teacher_included') is not False:
        raise ValueError('deployment-only checkpoint required')
    if checkpoint['config']!=c: raise ValueError('checkpoint configuration mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise ValueError('one authorized GPU required')
    out=Path(args.output)
    if out.exists(): raise ValueError('output already exists')
    model=CTSystem(c['residual_scale_hu'],c['prior_mode'],c['backbone']).cuda()
    model.load_state_dict(checkpoint['system'])
    dataset=make_dataset(c,'validation',out.with_suffix('.pixel_access.jsonl'))
    result=evaluate(model,dataset,patch_streams=c.get("validation_patch_streams",1))
    result.update({'checkpoint_sha256':sha256(args.checkpoint),'manifest_sha256':c['manifest_sha256']})
    with out.open('x') as f: json.dump(result,f,indent=2)


if __name__=='__main__':main()
