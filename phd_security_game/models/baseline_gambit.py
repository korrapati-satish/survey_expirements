from __future__ import annotations

import numpy as np


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.size, 1.0 / clipped.size, dtype=float)
    return clipped / total


def solve_gambit_nash_baseline(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Baseline model using PyGambit Nash equilibrium on a normal-form game.

    This is a simultaneous-move baseline (non-commitment), useful to compare against
    leader-follower Stackelberg and evolutionary adaptation.

    Selection rule when multiple equilibria exist:
    - Choose the equilibrium maximizing defender expected payoff.

    Returns:
        defender_sigma: defender mixed strategy
        attacker_sigma: attacker mixed strategy
        defender_value: expected defender payoff at selected equilibrium
    """
    import pygambit as gbt  # type: ignore

    d = np.asarray(defender_payoff, dtype=float)
    a = np.asarray(attacker_payoff, dtype=float)
    if d.ndim != 2 or a.ndim != 2 or d.shape != a.shape:
        raise ValueError("defender_payoff and attacker_payoff must be 2D with identical shape")

    n_def, n_att = d.shape
    game = gbt.Game.from_arrays(d, a)

    mixed_result = gbt.nash.enummixed_solve(game)
    solutions = list(mixed_result.equilibria)
    if not solutions:
        pure_result = gbt.nash.enumpure_solve(game)
        solutions = list(pure_result.equilibria)
    if not solutions:
        raise RuntimeError("PyGambit returned no Nash equilibrium for baseline model")

    players = list(game.players)
    defender_player = players[0]
    attacker_player = players[1]

    best_value = -np.inf
    best_defender = np.full(n_def, 1.0 / n_def, dtype=float)
    best_attacker = np.full(n_att, 1.0 / n_att, dtype=float)

    for sol in solutions:
        try:
            defender_sigma = _normalize_simplex(
                np.array([float(sol[s]) for s in defender_player.strategies], dtype=float)
            )
            attacker_sigma = _normalize_simplex(
                np.array([float(sol[s]) for s in attacker_player.strategies], dtype=float)
            )
        except Exception:
            continue

        value = float(defender_sigma @ d @ attacker_sigma)
        if value > best_value + 1e-12:
            best_value = value
            best_defender = defender_sigma
            best_attacker = attacker_sigma

    if not np.isfinite(best_value):
        raise RuntimeError("Failed to parse PyGambit equilibrium into baseline mixed strategies")

    return best_defender, best_attacker, float(best_value)
