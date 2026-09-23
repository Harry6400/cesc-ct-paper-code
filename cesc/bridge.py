from dataclasses import dataclass
from abc import ABC,abstractmethod
import torch
from torch import Tensor

@dataclass
class Batch:
    center_hu:Tensor
    b0_hu:Tensor
    target_hu:Tensor
    patient_ids:list[str]
    slice_ids:list[str]
    def validate(self,micro_batch:int|None=None):
        shape=self.center_hu.shape
        if len(shape)!=4 or shape[1]!=1 or min(shape[-2:])<4 or shape[-1]%4 or shape[-2]%4:
            raise ValueError('Batch must be [N,1,H,W] with H,W positive multiples of 4.')
        if micro_batch and shape[0]>micro_batch:raise ValueError('Microbatch exceeds contract.')
        if len(self.patient_ids)!=shape[0] or len(self.slice_ids)!=shape[0]:raise ValueError('Metadata count mismatch.')
        for t in (self.center_hu,self.b0_hu,self.target_hu):
            if t.shape!=shape or t.dtype!=torch.float32 or t.requires_grad or not torch.isfinite(t).all():
                raise ValueError('All HU tensors must be same-shaped finite detached FP32.')
        return self
    def to(self,device):
        return Batch(self.center_hu.to(device),self.b0_hu.to(device),self.target_hu.to(device),
                     self.patient_ids,self.slice_ids)

class Bridge(ABC):
    @abstractmethod
    def contract(self)->dict:
        """Include source, B0 checkpoint SHA, manifest SHA, normalization and metric version."""
    @abstractmethod
    def train_batches(self,epoch:int):pass
    @abstractmethod
    def validation_batches(self):pass
    @abstractmethod
    def metrics(self,pred_hu:Tensor,target_hu:Tensor)->dict:
        """One slice at a time; MUST reuse original metric definition in native runs."""
    def assert_frozen_b0(self):
        """Override: hash native B0 weights/buffers, check eval and requires_grad=False."""
        raise NotImplementedError('Native B0 freeze audit is not implemented.')
