import torch
from .metrics import PatientMetrics,paired_delta
from .statistics import statistics_loss

@torch.no_grad()
def evaluate(model,bridge,device,alpha:float=1.0,indices=None,prediction_callback=None)->dict:
    was_training=model.training;model.eval()
    try:
        candidate=PatientMetrics();baseline=PatientMetrics()
        batches = bridge.validation_batches() if indices is None else bridge.validation_batches(indices=indices)
        for batch in batches:
            batch.validate();batch=batch.to(device)
            pred=model(batch.center_hu,batch.b0_hu,alpha=alpha)
            for i,(p,s) in enumerate(zip(batch.patient_ids,batch.slice_ids)):
                if prediction_callback is not None:
                    prediction_callback(p,s,pred[i:i+1].detach().cpu())
                candidate.add(p,s,bridge.metrics(pred[i:i+1],batch.target_hu[i:i+1]))
                baseline.add(p,s,bridge.metrics(batch.b0_hu[i:i+1],batch.target_hu[i:i+1]))
        return paired_delta(candidate,baseline)
    finally:model.train(was_training)

@torch.no_grad()
def evaluate_statistics(model,bridge,device)->dict:
    was_training=model.training;model.eval()
    try:
        metrics=PatientMetrics()
        for batch in bridge.validation_batches():
            batch.validate();batch=batch.to(device)
            for i,(p,s) in enumerate(zip(batch.patient_ids,batch.slice_ids)):
                m=model.statistic_metric(batch.center_hu[i:i+1],batch.b0_hu[i:i+1])
                _,info=statistics_loss(m,(batch.target_hu[i:i+1]-batch.b0_hu[i:i+1])/model.cfg.s_ct)
                metrics.add(p,s,info)
        return metrics.summary()
    finally:model.train(was_training)
