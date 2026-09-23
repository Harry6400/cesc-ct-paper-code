# Third-party code

`native_baseline/src/ct_prior/official_restormer.py` is adapted from the official
[Restormer implementation](https://github.com/swz30/Restormer) by Syed Waqas Zamir
and contributors (CVPR 2022). It is distributed under the upstream MIT license,
reproduced in `RESTORMER_LICENSE.md`. The CT-specific five-slice backbone and
CESC-CT correction code are separate adaptations in this repository.

PyTorch, NumPy, einops, TensorBoard, and other installed dependencies retain
their own licenses. No pretrained Restormer or CT weights are distributed here.
