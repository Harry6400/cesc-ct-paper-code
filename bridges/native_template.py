"""Integration point only. This is deliberately NOT a fake implementation of the native project.

Implement NativeBridge using existing project imports; never replace its B0 or metrics.
See docs/NATIVE_INTEGRATION_zh.md for required provenance and tests.
"""
from cesc.bridge import Bridge

class NativeBridge(Bridge):
    def __init__(self,config:dict):
        raise NotImplementedError(
            'Native B0 source/checkpoint/dataset not present in this delivery. '
            'Implement this bridge locally, run preflight and authorize patient training explicitly.')
    def contract(self):raise NotImplementedError
    def train_batches(self,epoch):raise NotImplementedError
    def validation_batches(self):raise NotImplementedError
    def metrics(self,pred_hu,target_hu):raise NotImplementedError
    def assert_frozen_b0(self):raise NotImplementedError
