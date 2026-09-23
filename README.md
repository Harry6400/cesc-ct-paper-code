# CESC-CT: Conditional Error-Statistics Correction for Low-Dose CT

This repository contains the code for the CESC-CT v31 method described in the
accompanying ICASSP 2027 submission. It freezes a five-slice B0 restoration
backbone, trains a local error-statistics network, and then trains a three-step
shared-energy correction module. The Full, Diagonal, and Plain variants are in
`cesc/`; `bridges/native_e07_b0.py` connects the Full method to the registered
native B0 and Mayo development split. The native bridge is implemented; it is
not the synthetic fixture.

This is a **code and protocol release**. It contains no CT pixels, patient
volumes, model checkpoints, or result logs. The original frozen B0 deployment
checkpoint is required for exact paper reproduction and is not distributed.
The released code alone therefore does not reproduce the paper's reported
numbers. Synthetic runs test software wiring only and are not medical results.

## Files

- `cesc/`: statistics, conditional metric, energy correction, losses,
  checkpointing, and metrics.
- `bridges/native_e07_b0.py`: the native five-slice B0 and Mayo train/validation
  bridge used in the formal run; it has no locked-test loader.
- `native_baseline/src/`: native B0 architecture, preprocessing, and validation
  implementation required by that bridge.
- `scripts/run_ddp.py`: two-GPU formal training entry point.
- `scripts/infer_slice.py`: one-slice Full inference from either five quarter-dose
  slices plus the B0 checkpoint, or a precomputed B0 prediction.
- `data/mayo_manifest.example.json` and `configs/*.example.json`: path templates,
  not data or executable evidence.

## Environment and software smoke test

Use Python 3.10 or newer and install a PyTorch build suitable for the local
hardware. The formal runner requires two CUDA GPUs with NCCL; the synthetic
smoke test runs on CPU. From this repository's root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/synthetic_smoke.py --out outputs/synthetic_smoke
```

The smoke test trains tiny synthetic statistics, Full, Diagonal, and Plain
fixtures. It writes only beneath `outputs/` and does not use Mayo images.

## Data and frozen B0

Obtain the Mayo low-dose CT data from the
[TCIA LDCT collection](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/)
under its current access terms. Do not redistribute patient images in this
repository. The native loader expects paired quarter-dose and full-dose,
physically aligned 3 mm B30 volumes as NumPy arrays in HU, one `[Z,512,512]`
array per patient. Preserve physical slice order from DICOM geometry; folder
names or nominal SliceThickness alone do not define voxel spacing. Converting
raw DICOM into these aligned arrays is dataset preparation and is not supplied
here.

Copy `data/mayo_manifest.example.json` to `data/mayo_manifest.json` and replace
each `qd_path` and `fd_path` with the local aligned volume paths. The example
contains the paper's six training and three validation identities and expected
shapes. The held-out test patient is absent from the development manifest and
is never loaded by the training bridge. Adjust paths only; if the local data
have different shape, series, or geometry, they are not the paper's fixed
experiment. The data paths in the example are placeholders.

For the paper protocol, supply the **same frozen E07-origin B0
validation-best epoch-190 deployment checkpoint** used in the experiment. Its
SHA-256 is recorded in `contracts/paper_protocol.json`. The matching training
checkpoint SHA-256 is in both example configurations. A new B0 trained from
scratch is a different experiment and cannot reproduce the reported values.

## Two-stage training

Copy the two example configurations to `configs/statistics.json` and
`configs/full.json`. Set their `manifest_path`, `fixed_b0_checkpoint`, and
`pixel_audit_dir` to local paths; retain the paper's model and training
parameters. Both configs must use the same B0 checkpoint and manifest. Run
these commands only on authorized hardware with the appropriate dataset and
B0 weights:

```bash
python scripts/preflight.py --config configs/statistics.json --bridge bridges.native_e07_b0:NativeE07B0Bridge --device cuda:0 --out outputs/preflight.json
torchrun --standalone --nproc_per_node=2 scripts/run_ddp.py --config configs/statistics.json --bridge bridges.native_e07_b0:NativeE07B0Bridge --preflight-report outputs/preflight.json --contract contracts/paper_protocol.json --out outputs/statistics
torchrun --standalone --nproc_per_node=2 scripts/run_ddp.py --config configs/full.json --bridge bridges.native_e07_b0:NativeE07B0Bridge --preflight-report outputs/preflight.json --contract contracts/paper_protocol.json --stats-checkpoint outputs/statistics/best.pt --out outputs/full
```

Stage I selects the statistics checkpoint on validation data after 20 epochs.
Stage II freezes it and trains the Full correction for 200 epochs, validating
every 30 epochs. The global effective batch is 32 with two ranks, micro-batch
4 per rank, and accumulation 4. The output includes the Full training
checkpoint and `outputs/full/best_deployment.pt`. Do not select checkpoints on
the held-out test patient.

## Inference

With a trained Full deployment checkpoint and a trusted native B0 checkpoint,
one registered-size input can be processed end to end:

```bash
python scripts/infer_slice.py --qd-five-hu data/example_five_slices.npy --b0-checkpoint weights/b0_e190_deployment.pt --deployment outputs/full/best_deployment.pt --out outputs/prediction_hu.npy --device cuda:0
```

`example_five_slices.npy` is a `[5,512,512]` HU array centered on the desired
slice. The script verifies the B0 checkpoint SHA-256 against the Full
deployment provenance. For an existing B0 prediction, supply two `[H,W]` HU
arrays instead:

```bash
python scripts/infer_slice.py --center-hu data/center_hu.npy --b0-hu data/b0_hu.npy --deployment outputs/full/best_deployment.pt --out outputs/prediction_hu.npy
```

The second form assumes the B0 array was generated by the frozen paper
backbone. The script writes one HU image in NumPy format. Load checkpoints
only from trusted sources because PyTorch checkpoint loading uses pickle.

## Provenance and licenses

The paper uses the frozen B0 selected on validation, the Mayo patient-level
6/3/1 split, a 20-epoch statistics stage, and a 200-epoch Full stage. No
LIDC claim or locked-test selection is encoded in this release. Method code is
under `LICENSE`. The native backbone includes a file adapted from the official
Restormer implementation; see `THIRD_PARTY_NOTICES.md` and
`RESTORMER_LICENSE.md` for attribution and its MIT terms.
