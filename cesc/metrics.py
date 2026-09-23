"""Patient-first aggregation. Numerical metric implementation belongs to the native bridge."""
from collections import defaultdict
import math
from statistics import mean

class PatientMetrics:
    def __init__(self):self.rows={}
    def add(self,patient_id:str,slice_id:str,values:dict):
        key=(patient_id,slice_id)
        if key in self.rows:raise ValueError(f'Duplicate slice key: {key}')
        if not values or any(not math.isfinite(float(v)) for v in values.values()):
            raise ValueError('Empty/nonfinite metric dictionary.')
        if self.rows and set(values)!=set(next(iter(self.rows.values()))):
            raise ValueError('Inconsistent metric keys.')
        self.rows[key]={k:float(v) for k,v in values.items()}
    def summary(self)->dict:
        if not self.rows:raise ValueError('Cannot aggregate empty validation.')
        by_patient=defaultdict(list)
        for (p,_),v in self.rows.items():by_patient[p].append(v)
        patient={p:{k:mean(r[k] for r in rows) for k in rows[0]} for p,rows in by_patient.items()}
        overall={k:mean(v[k] for v in patient.values()) for k in next(iter(patient.values()))}
        return {'patient_equal_mean':overall,'patients':patient,'slice_count':len(self.rows)}

def paired_delta(candidate:PatientMetrics,baseline:PatientMetrics)->dict:
    if candidate.rows.keys()!=baseline.rows.keys():raise ValueError('Candidate/B0 slice-key mismatch.')
    c=candidate.summary();b=baseline.summary()
    return {'candidate':c,'fixed_b0':b,
            'slice_metrics':[{'patient_id':p,'slice_id':s,'candidate':candidate.rows[(p,s)],'fixed_b0':baseline.rows[(p,s)]} for p,s in sorted(candidate.rows)],
            'delta':{k:c['patient_equal_mean'][k]-b['patient_equal_mean'][k]
                     for k in c['patient_equal_mean']}}
