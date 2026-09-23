"""Run four synthetic training roles. No metric in this output is a patient result."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cesc.runner import run
p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/demo.json').read_text())
s=copy.deepcopy(cfg);s['stage']='statistics'
run(s,'bridges.synthetic:SyntheticBridge',root/'statistics')
for role in ('full','diagonal','plain'):
    c=copy.deepcopy(cfg);c['variant']=role
    run(c,'bridges.synthetic:SyntheticBridge',root/role,
        stats_checkpoint=str(root/'statistics/best.pt') if role!='plain' else None)
print('SYNTHETIC ONLY: statistics, full, diagonal, plain finished. Not evidence of CT efficacy.')
