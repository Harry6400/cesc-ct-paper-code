"""Mechanism diagnostics, separate from registered training ablations."""
import torch


def shuffled_prior(prior, generator):
    permutation=torch.randperm(prior.shape[-2]*prior.shape[-1],generator=generator,device=prior.device)
    return prior.flatten(2)[:,:,permutation].reshape_as(prior)


@torch.no_grad()
def prior_diagnostic(objective,y,target,noise,generator):
    s=objective.system
    teacher=objective.teacher_prior(y,target)
    predicted=s.predict_prior(y,noise)
    base=s.restorer.reconstruct(y,None,s.scale_hu)
    restored=s.restorer.reconstruct(y,s.read_prior(predicted),s.scale_hu)
    shuffled=s.restorer.reconstruct(y,s.read_prior(shuffled_prior(predicted,generator)),s.scale_hu)
    return {'role':'teacher_mechanism_diagnostic_not_formal_result','prior_l1':float((predicted-teacher).abs().mean()),
            'base_mae_hu':float((base-target).abs().mean()),'predicted_mae_hu':float((restored-target).abs().mean()),
            'shuffled_mae_hu':float((shuffled-target).abs().mean())}
