"""Read-only decision from existing validation JSON; never launches experiments."""
import argparse
import json
from pathlib import Path
from .gates import qualify


def assess(method, controls, dataset):
    checks={name:qualify(method,value) for name,value in controls.items()}
    for name,c in controls.items():
        for key in ('manifest_sha256','s0_sha256','stage','step','kind'):
            if method.get(key) is None or method.get(key)!=c.get(key):
                checks[name]['passed']=False;checks[name]['failures'].append('unmatched '+key)
    m=method['metrics']
    target=(m['mae_hu']<=9.955918 and m['psnr_db']>=48.227881 and m['ssim']>=.985275) if dataset=='mayo' else None
    return {'checks':checks,'mechanism_gate_passed':all(c['passed'] for c in checks.values()),
            'mayo_total_target_met':target,'efficacy_claim_requires_real_evidence':True,
            'high_density_review':{name:{p:v.get('high_density_mae_hu') for p,v in c['patients'].items()} for name,c in {'M':method,**controls}.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--b0',required=True);p.add_argument('--deterministic',required=True);p.add_argument('--dataset',choices=['mayo','lidc'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    read=lambda path:json.loads(Path(path).read_text())
    result=assess(read(a.method),{'B0':read(a.b0),'D':read(a.deterministic)},a.dataset)
    with Path(a.output).open('x') as f: json.dump(result,f,indent=2)
if __name__=='__main__':main()
