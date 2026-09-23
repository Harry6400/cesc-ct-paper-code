from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EarlyStopState:
    """The same two-stage state machine used by the v10n LIDC parent."""

    best_psnr: float = -math.inf
    best_mae: float = math.inf
    significant_anchor: float = -math.inf
    no_significant_checks: int = 0
    lr_reduced: bool = False
    recovered_after_reduce: bool = False
    pre_reduce_best: float | None = None
    checks_after_reduce: int = 0
    consecutive_divergence: int = 0
    epoch20_gate_enabled: bool = True
    history: list[tuple[int, float]] = field(default_factory=list)

    @staticmethod
    def _slope(values: list[tuple[int, float]]) -> float:
        if len(values) < 4:
            return math.inf
        y = np.asarray([value for _, value in values[-4:]], dtype=np.float64)
        return float(np.polyfit(np.arange(4, dtype=np.float64), y, 1)[0])

    def update(self, epoch: int, metrics: dict[str, float]) -> dict:
        psnr, mae, ssim = (float(metrics[key]) for key in ("psnr_hu", "mae_hu", "ssim_hu"))
        input_psnr, input_mae, input_ssim = (
            float(metrics[key]) for key in ("input_psnr_hu", "input_mae_hu", "input_ssim_hu")
        )
        event = {"epoch": int(epoch), "action": "continue", "reason": "none"}
        if not all(math.isfinite(value) for value in (psnr, mae, ssim)):
            return {**event, "action": "stop", "reason": "nonfinite_validation"}
        divergent = (
            math.isfinite(self.best_psnr)
            and psnr <= self.best_psnr - 0.30
            and mae >= self.best_mae + 2.0
        )
        self.consecutive_divergence = self.consecutive_divergence + 1 if divergent else 0
        if self.consecutive_divergence >= 2:
            return {**event, "action": "stop", "reason": "two_check_divergence"}
        if psnr > self.best_psnr:
            self.best_psnr, self.best_mae = psnr, mae
        if psnr >= self.significant_anchor + 0.02:
            self.significant_anchor, self.no_significant_checks = psnr, 0
        else:
            self.no_significant_checks += 1
        self.history.append((int(epoch), psnr))
        if self.epoch20_gate_enabled and epoch == 20 and not (
            psnr - input_psnr >= 1.0 and mae - input_mae <= -10.0 and ssim >= input_ssim
        ):
            return {**event, "action": "stop", "reason": "epoch20_convergence_gate"}
        if epoch >= 200:
            return {**event, "action": "stop", "reason": "epoch200_cap"}
        if self.lr_reduced and not self.recovered_after_reduce:
            self.checks_after_reduce += 1
            if self.pre_reduce_best is not None and psnr >= self.pre_reduce_best + 0.01:
                self.recovered_after_reduce = True
                self.no_significant_checks = 0
                self.significant_anchor = psnr
                return {**event, "reason": "post_reduce_recovery"}
            if self.checks_after_reduce >= 4:
                return {**event, "action": "stop", "reason": "no_post_reduce_recovery"}
        plateau = epoch >= 40 and self.no_significant_checks >= 6 and self._slope(self.history) <= 0.005
        if plateau:
            if not self.lr_reduced:
                self.lr_reduced = True
                self.pre_reduce_best = self.best_psnr
                self.checks_after_reduce = 0
                return {**event, "action": "reduce_lr", "reason": "first_plateau"}
            if self.recovered_after_reduce:
                return {**event, "action": "stop", "reason": "second_plateau"}
        return event
