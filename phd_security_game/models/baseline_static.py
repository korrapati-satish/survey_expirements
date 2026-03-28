from __future__ import annotations

import numpy as np


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = float(clipped.sum())
    if s <= 0.0:
        return np.full(clipped.size, 1.0 / clipped.size, dtype=float)
    return clipped / s


def solve_static_defender_baseline(
    defender_payoff: np.ndarray,
    attacker_reference_strategy: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Static (non-adaptive) defender baseline.

    The defender chooses a single fixed pure strategy once, based on a reference
    attacker distribution, and reuses it for all episodes.

    Objective:
        i* = argmax_i U_D(i, :) · p_A

    Returns:
        defender_sigma: one-hot pure static defender strategy
        value: expected defender payoff under reference attacker mix
    """
    d = np.asarray(defender_payoff, dtype=float)
    p_a = _normalize_simplex(np.asarray(attacker_reference_strategy, dtype=float))

    if d.ndim != 2:
        raise ValueError("defender_payoff must be a 2D matrix")
    if p_a.ndim != 1 or p_a.size != d.shape[1]:
        raise ValueError("attacker_reference_strategy shape must match defender_payoff columns")

    expected = d @ p_a
    best_idx = int(np.argmax(expected))

    defender_sigma = np.zeros(d.shape[0], dtype=float)
    defender_sigma[best_idx] = 1.0
    return defender_sigma, float(expected[best_idx])
