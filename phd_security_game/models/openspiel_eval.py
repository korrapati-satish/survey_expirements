from __future__ import annotations

import numpy as np


def evaluate_with_openspiel_rl(
    defender_payoff: np.ndarray,
    attacker_strategy: np.ndarray,
    episodes: int,
    seed: int,
    epsilon: float = 0.10,
    alpha: float = 0.20,
) -> float:
    """RL-based strategy evaluation using OpenSpiel matrix game wrapper."""
    import pyspiel  # type: ignore

    _ = pyspiel.create_matrix_game(defender_payoff.tolist(), (-defender_payoff).tolist())

    rng = np.random.default_rng(seed)
    n_def, n_att = defender_payoff.shape
    q = np.zeros((n_att, n_def), dtype=float)

    for _ in range(episodes):
        att = int(rng.choice(n_att, p=attacker_strategy))
        if rng.random() < epsilon:
            deff = int(rng.integers(0, n_def))
        else:
            deff = int(np.argmax(q[att]))

        reward = float(defender_payoff[deff, att])
        q[att, deff] = q[att, deff] + alpha * (reward - q[att, deff])

    learned_def = np.zeros(n_def, dtype=float)
    for att in range(n_att):
        learned_def[int(np.argmax(q[att]))] += attacker_strategy[att]

    if learned_def.sum() > 0.0:
        learned_def /= learned_def.sum()
    else:
        learned_def[:] = 1.0 / n_def

    return float(learned_def @ defender_payoff @ attacker_strategy)
