import torch
from .losses import reconstruction_loss
from .statistics import statistics_loss
from .audit import optimizer_coverage

def train_epoch(model,bridge,optimizer,epoch:int,accumulation:int,device,micro_batch:int)->dict:
    if accumulation<1:raise ValueError('accumulation must be positive.')
    model.train();optimizer_coverage(model,optimizer)
    optimizer.zero_grad(set_to_none=True)
    count=0;total=0.0;steps=0;group_samples=0;group_batches=0
    # Backpropagate sample-weighted means then divide gradients by actual samples.
    # This treats a partial final accumulation group correctly.
    for batch in bridge.train_batches(epoch):
        batch.validate(micro_batch);batch=batch.to(device);n=len(batch.patient_ids)
        if model.stage=='statistics':
            metric=model.statistic_metric(batch.center_hu,batch.b0_hu)
            loss,_=statistics_loss(metric,(batch.target_hu-batch.b0_hu)/model.cfg.s_ct)
        elif model.stage=='correction':
            pred=model(batch.center_hu,batch.b0_hu)
            loss=reconstruction_loss(pred,batch.target_hu,model.cfg.s_ct)
        else:raise RuntimeError('Model stage must be statistics or correction.')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss.')
        (loss*n).backward();total+=float(loss.detach())*n;count+=n
        group_samples+=n;group_batches+=1
        if group_batches==accumulation:
            _step(model,optimizer,group_samples);steps+=1;group_samples=group_batches=0
    if group_samples:_step(model,optimizer,group_samples);steps+=1
    if not count:raise RuntimeError('Training bridge returned no samples.')
    bridge.assert_frozen_b0()
    return {'loss':total/count,'samples':count,'optimizer_steps':steps}

def _step(model,optimizer,samples):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(samples)
            if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient.')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
