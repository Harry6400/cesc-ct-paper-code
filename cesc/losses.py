from torch import Tensor

def reconstruction_loss(pred_hu:Tensor,target_hu:Tensor,s_ct:float)->Tensor:
    error=(pred_hu-target_hu)/s_ct
    return error.abs().mean()+0.1*error.square().mean()
