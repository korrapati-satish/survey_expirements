from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SecurityGameConfig:
    defender_strategies: Sequence[str]
    attacker_actions: Sequence[str]
    is_attack_action: Sequence[bool]
    defender_payoff: np.ndarray
    attacker_payoff: np.ndarray
    detection_probability: np.ndarray

    def validate(self) -> None:
        d_count = len(self.defender_strategies)
        a_count = len(self.attacker_actions)

        if len(self.is_attack_action) != a_count:
            raise ValueError("is_attack_action must have one boolean per attacker action")

        expected_shape = (d_count, a_count)
        for name, matrix in (
            ("defender_payoff", self.defender_payoff),
            ("attacker_payoff", self.attacker_payoff),
            ("detection_probability", self.detection_probability),
        ):
            if matrix.shape != expected_shape:
                raise ValueError(f"{name} shape {matrix.shape} must match {expected_shape}")

        if np.any(self.detection_probability < 0.0) or np.any(self.detection_probability > 1.0):
            raise ValueError("detection_probability entries must be in [0, 1]")


@dataclass(frozen=True)
class ExperimentResult:
    defender_stackelberg_strategy: np.ndarray
    attacker_stackelberg_strategy: np.ndarray
    defender_evolutionary_strategy: np.ndarray
    attacker_evolutionary_strategy: np.ndarray

    # Legacy primary metrics (kept for backward compatibility; stackelberg-based)
    payoff_gain: float
    attack_detection_rate: float
    false_positive_rate: float

    # Explicit model comparison metrics
    stackelberg_payoff_gain: float
    evolutionary_payoff_gain: float
    payoff_gain_delta_evo_minus_stack: float

    stackelberg_attack_detection_rate: float
    evolutionary_attack_detection_rate: float
    adr_delta_evo_minus_stack: float

    stackelberg_false_positive_rate: float
    evolutionary_false_positive_rate: float
    fpr_delta_evo_minus_stack: float

    convergence_time_seconds: float
    convergence_iterations: int
    gambit_normal_form_equilibria: int
    gambit_extensive_form_equilibria: int
    openspiel_rl_payoff_estimate: Optional[float]
    openspiel_rl_payoff_estimate_stackelberg: Optional[float]
    openspiel_rl_payoff_estimate_evolutionary: Optional[float]

    # Optional simultaneous-move baseline (PyGambit Nash)
    baseline_payoff_gain: Optional[float] = None
    baseline_attack_detection_rate: Optional[float] = None
    baseline_false_positive_rate: Optional[float] = None

    # Optional static fixed-policy baseline
    static_baseline_payoff_gain: Optional[float] = None
    static_baseline_attack_detection_rate: Optional[float] = None
    static_baseline_false_positive_rate: Optional[float] = None

    def to_frame(self) -> pd.DataFrame:
        metrics = [
            ("Stackelberg Payoff Gain (Defender Net Utility)", self.stackelberg_payoff_gain),
            ("Evolutionary Payoff Gain (Defender Net Utility)", self.evolutionary_payoff_gain),
            ("Delta Payoff (Evo - Stack)", self.payoff_gain_delta_evo_minus_stack),
            ("Stackelberg ADR (%)", self.stackelberg_attack_detection_rate),
            ("Evolutionary ADR (%)", self.evolutionary_attack_detection_rate),
            ("Delta ADR (Evo - Stack, %)", self.adr_delta_evo_minus_stack),
            ("Stackelberg FPR (%)", self.stackelberg_false_positive_rate),
            ("Evolutionary FPR (%)", self.evolutionary_false_positive_rate),
            ("Delta FPR (Evo - Stack, %)", self.fpr_delta_evo_minus_stack),
            ("Convergence Time (s)", self.convergence_time_seconds),
            ("Convergence Iterations", self.convergence_iterations),
            ("Gambit Normal-Form Equilibria", self.gambit_normal_form_equilibria),
            ("Gambit Extensive-Form Equilibria", self.gambit_extensive_form_equilibria),
        ]
        if self.openspiel_rl_payoff_estimate_stackelberg is not None:
            metrics.append(("OpenSpiel RL Payoff Estimate (Stackelberg)", self.openspiel_rl_payoff_estimate_stackelberg))
        if self.openspiel_rl_payoff_estimate_evolutionary is not None:
            metrics.append(("OpenSpiel RL Payoff Estimate (Evolutionary)", self.openspiel_rl_payoff_estimate_evolutionary))
        if self.baseline_payoff_gain is not None:
            metrics.append(("Baseline Nash Payoff Gain", self.baseline_payoff_gain))
        if self.baseline_attack_detection_rate is not None:
            metrics.append(("Baseline Nash ADR (%)", self.baseline_attack_detection_rate))
        if self.baseline_false_positive_rate is not None:
            metrics.append(("Baseline Nash FPR (%)", self.baseline_false_positive_rate))
        if self.static_baseline_payoff_gain is not None:
            metrics.append(("Baseline Static Payoff Gain", self.static_baseline_payoff_gain))
        if self.static_baseline_attack_detection_rate is not None:
            metrics.append(("Baseline Static ADR (%)", self.static_baseline_attack_detection_rate))
        if self.static_baseline_false_positive_rate is not None:
            metrics.append(("Baseline Static FPR (%)", self.static_baseline_false_positive_rate))

        return pd.DataFrame({"metric": [m[0] for m in metrics], "value": [m[1] for m in metrics]})
