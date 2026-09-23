"""Evidence/resume guards. No automatic scheduling, stopping, or remote writes."""
from __future__ import annotations
import math
from collections import defaultdict
from .common import json_digest

RESUME_FIELDS=('schema','variant','architecture_sha256','native_source_sha256',
               'manifest_sha256','objective_sha256','optimizer_config_sha256',
               'scheduler_config_sha256','world_size','micro_batch','accumulation',
               'precision','native_seed','native_initial_sha256','data_order_contract_sha256',
               'historical_lr_trace_sha256','extension_source_sha256','group_lr_mapping_sha256')

def assert_resume_compatible(stored:dict,current:dict)->None:
    for key in RESUME_FIELDS:
        if key not in stored or key not in current or stored[key] is None or current[key] is None:
            raise ValueError(f'Missing resume identity field: {key}')
        if stored[key]!=current[key]:
            raise ValueError(f'Resume identity changed: {key}')
    if current['schema']!='ct_v29_r2_20260921':
        raise ValueError('Not a v29-r2 checkpoint; no implicit migration from v27/v28/B0.')

def patient_equal_delta(candidate_rows:list[dict],reference_rows:list[dict],metric:str)->dict:
    """Strict same-(patient,slice) comparison. No inferred or intersect-only pairing."""
    def index(rows):
        out={}
        for row in rows:
            key=(str(row['patient_id']),str(row['slice_id']))
            if key in out: raise ValueError(f'Duplicate slice key: {key}')
            if metric not in row or not math.isfinite(float(row[metric])):
                raise ValueError(f'Missing/nonfinite {metric} at {key}')
            out[key]=float(row[metric])
        return out
    c,r=index(candidate_rows),index(reference_rows)
    if not c or c.keys()!=r.keys():
        raise ValueError('Missing/extra slice keys; cannot silently intersect or synthesize reference rows.')
    grouped=defaultdict(list)
    for key,value in c.items(): grouped[key[0]].append(value-r[key])
    patients={p:sum(v)/len(v) for p,v in sorted(grouped.items())}
    return {'metric':metric,'patient_deltas':patients,
            'patient_equal_delta':sum(patients.values())/len(patients),
            'patients':len(patients),'slices':len(c),
            'reference_role':'historical_B0','is_independent_patient_replication':False}

def classify_window(rows:list[dict],required=(90,100,110,120,130))->dict:
    """Budget-review suggestion ONLY; exact raw rows, not screenshot-rounded values."""
    by_epoch={}
    for row in rows:
        e=int(row['epoch'])
        if e in by_epoch: raise ValueError('Duplicate epoch')
        if row.get('source_quality')!='raw_validation_json':
            raise ValueError('Window gate requires raw JSON, not screenshot transcriptions.')
        by_epoch[e]=row
    missing=[e for e in required if e not in by_epoch]
    if missing: return {'status':'INCOMPLETE','missing_epochs':missing,'automatic_stop':False}
    ps=[float(by_epoch[e]['delta_psnr_db']) for e in required]
    ma=[float(by_epoch[e]['delta_mae_hu']) for e in required]
    if not all(math.isfinite(v) for v in ps+ma): raise ValueError('Nonfinite window values')
    negative=sum(v<0 for v in ps)
    review=sum(ps)/len(ps)<=0 and negative>=max(1,len(required)-1)
    return {'status':'BUDGET_REVIEW' if review else 'CONTINUE_OR_REVIEW',
            'mean_delta_psnr_db':sum(ps)/len(ps),'mean_delta_mae_hu':sum(ma)/len(ma),
            'psnr_negative_points':negative,'automatic_stop':False,
            'psnr_primary_review_not_blocked_by_mae':True,
            'not_a_statistical_significance_test':True,
            'not_proof_of_permanent_failure':True}


def select_validation_best(rows:list[dict])->dict:
    """Absolute patient-equal validation metrics only. Not max(candidate-B0)."""
    if not rows: raise ValueError('Empty validation records.')
    epochs=set()
    for row in rows:
        if row.get('split')!='validation' or row.get('source_quality')!='raw_validation_json':
            raise ValueError('Select checkpoints using raw validation-only records.')
        if row['epoch'] in epochs: raise ValueError('Duplicate validation epoch')
        epochs.add(row['epoch'])
        for metric in ('psnr','ssim','mae'):
            if metric not in row or not math.isfinite(float(row[metric])):
                raise ValueError('Missing/nonfinite absolute metric')
    return max(rows,key=lambda x:(float(x['psnr']),float(x['ssim']),-float(x['mae']),-int(x['epoch'])))
