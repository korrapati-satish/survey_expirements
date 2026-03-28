from __future__ import annotations

import time

import numpy as np

from phd_security_game.config import EvolutionaryConfig


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size)
    return clipped / total


def _run_nashpy_replicator(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
    defender_x0: np.ndarray,
    attacker_y0: np.ndarray,
    cfg: EvolutionaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    import nashpy as nash  # type: ignore

    game = nash.Game(defender_payoff, attacker_payoff)
    timepoints = np.linspace(0.0, float(cfg.max_iterations), num=cfg.max_iterations + 1)
    x_traj, y_traj = game.asymmetric_replicator_dynamics(
        x0=defender_x0,
        y0=attacker_y0,
        timepoints=timepoints,
    )
    return np.asarray(x_traj, dtype=float), np.asarray(y_traj, dtype=float)


def run_evolutionary_dynamics(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
    cfg: EvolutionaryConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Evolutionary game adaptation using Nashpy replicator dynamics.
    Both populations co-evolve from a uniform distribution to find a stable Nash Equilibrium.
    """
    start = time.perf_counter()

    n_def = defender_payoff.shape[0]
    n_att = attacker_payoff.shape[1]
    
    x0 = np.full(n_def, 1.0 / n_def, dtype=float)
    y0 = np.full(n_att, 1.0 / n_att, dtype=float)

    defender_traj, attacker_traj = _run_nashpy_replicator(
        defender_payoff=defender_payoff,
        attacker_payoff=attacker_payoff,
        defender_x0=x0,
        attacker_y0=y0,
        cfg=cfg,
    )

    traj_x = np.array([_normalize_simplex(row) for row in defender_traj], dtype=float)
    traj_y = np.array([_normalize_simplex(row) for row in attacker_traj], dtype=float)

    stable = 0
    for it in range(1, traj_y.shape[0]):
        delta_x = float(np.linalg.norm(traj_x[it] - traj_x[it - 1], ord=1))
        delta_y = float(np.linalg.norm(traj_y[it] - traj_y[it - 1], ord=1))
        if delta_x < cfg.epsilon and delta_y < cfg.epsilon:
            stable += 1
            if stable >= cfg.stable_window:
                return traj_x[it], traj_y[it], float(time.perf_counter() - start), it
        else:
            stable = 0

    return traj_x[-1], traj_y[-1], float(time.perf_counter() - start), cfg.max_iterations
