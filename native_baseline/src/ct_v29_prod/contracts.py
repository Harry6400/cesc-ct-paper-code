"""Validation of the frozen v29-r2 production and authorization contract."""
from __future__ import annotations

from .model import FORMAL_VARIANTS, VARIANTS


def validate_run_contract(config: dict, *, mode: str) -> None:
    required = {
        'variant', 'seed', 'precision', 'residual_scale_hu', 'backbone',
        'primary_lr', 'optimizer_betas', 'optimizer_eps', 'weight_decay',
        'package', 'resolved_contract_path', 'manifest_path', 'lr_trace_path',
        'historical_b0_curve_path', 'output_dir', 'target_instance',
        'checkpoint_source', 'split', 'micro_batch', 'accumulation',
        'effective_batch', 'validation_interval_epochs', 'max_epochs',
        'validation_patch_batch', 'validation_pin_memory',
        'validation_metric_workers', 'validation_cuda_graph',
        'historical_reference_b0', 'explicit_gpu_smoke_authorization',
        'explicit_formal_run_authorization', 'authorized_gpu_uuid',
        'native_source_sha256', 'objective_sha256',
        'optimizer_config_sha256', 'scheduler_config_sha256',
        'data_order_contract_sha256',
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f'missing v29 contract fields: {missing}')
    if config['variant'] not in VARIANTS:
        raise ValueError('unknown v29 variant')
    if int(config['seed']) != 123456 or config['precision'] != 'fp32':
        raise ValueError('v29 requires native seed 123456 and FP32')
    if config['package'] != 'RoDiff_CT_v29_r2_noS_resolved':
        raise ValueError('wrong package identity')
    if config['checkpoint_source'] != 'fresh' or config['split'] != '6/3/1':
        raise ValueError('v29 requires fresh Mayo 6/3/1')
    if (int(config['micro_batch']), int(config['accumulation']),
            int(config['effective_batch'])) != (4, 8, 32):
        raise ValueError('effective batch contract changed')
    if int(config['max_epochs']) != 200:
        raise ValueError('training horizon changed')
    if config['variant'] == 'B05center':
        if config.get('context_mode') != 'center_repeat':
            raise ValueError('B05center requires center_repeat input')
        if int(config['validation_interval_epochs']) != 10:
            raise ValueError('B05center requires 10-epoch validation cadence')
        if config.get('probe_epochs') != [] or config.get('retain_epochs') != []:
            raise ValueError('B05center forbids probes and milestone checkpoints')
    elif config['variant'] == 'B0':
        if config.get('context_mode') != 'real_five':
            raise ValueError('B0 requires original real_five input')
        if int(config['validation_interval_epochs']) != 10:
            raise ValueError('E07 B0 replication requires 10-epoch validation cadence')
        if config.get('probe_epochs') != [] or config.get('retain_epochs') != []:
            raise ValueError('E07 B0 replication forbids probes and milestone checkpoints')
    else:
        if config.get('context_mode', 'real_five') != 'real_five':
            raise ValueError('existing variants require real_five input')
        if int(config['validation_interval_epochs']) != 30:
            raise ValueError('training cadence changed')
        if config.get('probe_epochs', [40, 90, 130]) != [40, 90, 130] or config.get('retain_epochs', [40, 90, 130]) != [40, 90, 130]:
            raise ValueError('existing probe/retention policy changed')
    if int(config['validation_patch_batch']) not in {1, 2, 4, 8, 16, 25}:
        raise ValueError('unsupported validation patch batch')
    if int(config['validation_metric_workers']) not in {0, 1, 2}:
        raise ValueError('unsupported metric worker count')
    reference = config['historical_reference_b0']
    if reference != {
        'paired_b0_gate': 'USER_WAIVED',
        'baseline_type': 'HISTORICAL_REFERENCE_B0',
        'matched_fresh_b0_available': False,
    }:
        raise ValueError('historical B0 declaration changed')
    if mode in {'smoke', 'profile'} and config['explicit_gpu_smoke_authorization'] is not True:
        raise PermissionError('GPU smoke/profile is not explicitly authorized')
    if mode in {'formal', 'resume'}:
        if config['variant'] not in FORMAL_VARIANTS:
            raise PermissionError('production formal permits only explicitly registered variants')
        if config['explicit_formal_run_authorization'] is not True:
            raise PermissionError('formal/resume is not explicitly authorized')
