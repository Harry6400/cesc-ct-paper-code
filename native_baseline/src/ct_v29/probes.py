"""Recomputed, evaluation-only mechanism probes; no prediction blending."""
from __future__ import annotations
import copy
from .model import probe_mode

def alpha_predictions(model,y,*,scale_hu=22.299220188880145):
    points=[('on',1.,1.,1.)]
    if model.a29 is not None:points.extend([('a_half',.5,1.,1.),('a_raw',0.,1.,1.)])
    if model.b29 is not None:points.extend([('b_half',1.,.5,1.),('b_mean',1.,0.,1.)])
    if model.carrier is not None:points.append(('extension_off',1.,1.,0.))
    predictions,records={},[]
    for label,aa,bb,gg in points:
        with probe_mode(model,aa,bb,alpha_extension=gg,diagnostics=True):
            predictions[label]=model.reconstruct(y,scale_hu=scale_hu).detach()
            records.append({'label':label,'alpha_a':aa,'alpha_b':bb,'alpha_extension':gg,
                'scope':'eval_only_recomputed_forward','model':copy.deepcopy(model.last_diagnostics),
                'a':copy.deepcopy(model.a29.last_diagnostics) if model.a29 else None,
                'b':copy.deepcopy(model.b29.last_diagnostics) if model.b29 else None,
                'eligible_for_checkpoint_selection':False,'off_is_coadapted_not_independent_B0':True})
    return predictions,records
