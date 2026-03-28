from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from typing import Literal


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.size, 1.0 / clipped.size, dtype=float)
    return clipped / total


def _validate_bimatrix(defender_payoff: np.ndarray, attacker_payoff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(defender_payoff, dtype=float)
    a = np.asarray(attacker_payoff, dtype=float)
    if d.ndim != 2 or a.ndim != 2:
        raise ValueError("defender_payoff and attacker_payoff must be 2D matrices")
    if d.shape != a.shape:
        raise ValueError(f"Payoff shape mismatch: {d.shape} vs {a.shape}")
    if d.shape[0] == 0 or d.shape[1] == 0:
        raise ValueError("Payoff matrices must be non-empty")
    return d, a


def _solve_stackelberg_lp(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve Stackelberg with an LP per attacker action.

    For each attacker action j, solve:
        max_x      U_D(:, j)^T x
        s.t.       x in simplex
                   U_A(:, j)^T x >= U_A(:, k)^T x for all k

    The best feasible j yields a strong Stackelberg equilibrium under optimistic tie-breaking.
    """
    n_def, n_att = defender_payoff.shape
    best_value = -np.inf
    best_x: np.ndarray | None = None
    best_j = 0

    bounds = [(0.0, 1.0)] * n_def
    A_eq = np.ones((1, n_def), dtype=float)
    b_eq = np.array([1.0], dtype=float)

    for j in range(n_att):
        # Attacker best-response constraints: U_A(:,j)^T x >= U_A(:,k)^T x for all k
        # linprog expects A_ub x <= b_ub, so we rewrite as:
        # (U_A(:,k)-U_A(:,j))^T x <= 0
        A_ub = []
        b_ub = []
        for k in range(n_att):
            if k == j:
                continue
            A_ub.append((attacker_payoff[:, k] - attacker_payoff[:, j]).astype(float))
            b_ub.append(0.0)

        c = -defender_payoff[:, j]  # maximize => minimize negative
        res = linprog(
            c=c,
            A_ub=np.array(A_ub, dtype=float) if A_ub else None,
            b_ub=np.array(b_ub, dtype=float) if b_ub else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success or res.x is None:
            continue

        x = _normalize_simplex(np.asarray(res.x, dtype=float))
        value = float(defender_payoff[:, j] @ x)
        if value > best_value + 1e-12:
            best_value = value
            best_x = x
            best_j = j

    if best_x is None:
        raise RuntimeError("No feasible Stackelberg commitment found for any attacker best-response action")

    attacker_sigma = np.zeros(n_att, dtype=float)
    attacker_sigma[best_j] = 1.0
    return best_x, attacker_sigma, float(best_value)


def _solve_stackelberg_pure_with_gambit_structure(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve pure-commitment Stackelberg while explicitly constructing a Gambit extensive-form game.

    Notes:
    - This path uses PyGambit to build the leader-follower game tree representation.
    - The solution step remains deterministic best-response enumeration for robustness.
    - This computes pure leader commitment, not mixed-leader SSE.
    """
    import pygambit as gbt  # type: ignore

    n_def, n_att = defender_payoff.shape

    # Build a leader-follower extensive-form representation for compatibility/inspection.
    game = gbt.Game.new_tree(players=["Defender", "Attacker"], title="Stackelberg Extensive Security Game")
    defender_actions = [f"D{i}" for i in range(n_def)]
    attacker_actions = [f"A{j}" for j in range(n_att)]
    game.append_move(game.root, "Defender", defender_actions)
    game.append_move(game.root.children, "Attacker", attacker_actions)
    for i in range(n_def):
        for j in range(n_att):
            outcome = game.add_outcome([float(defender_payoff[i, j]), float(attacker_payoff[i, j])])
            game.set_outcome(game.root.children[i].children[j], outcome)

    # Robust pure Stackelberg solve: leader pure commitment + follower pure best-response.
    best_payoff = -np.inf
    best_def_idx = 0
    best_att_idx = 0
    for def_idx in range(n_def):
        att_idx = int(np.argmax(attacker_payoff[def_idx]))
        def_payoff = float(defender_payoff[def_idx, att_idx])
        if def_payoff > best_payoff + 1e-12:
            best_payoff = def_payoff
            best_def_idx = def_idx
            best_att_idx = att_idx

    defender_sigma = np.zeros(n_def, dtype=float)
    defender_sigma[best_def_idx] = 1.0
    attacker_sigma = np.zeros(n_att, dtype=float)
    attacker_sigma[best_att_idx] = 1.0
    return defender_sigma, attacker_sigma, float(best_payoff)


def solve_stackelberg_security_game(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
    method: Literal["lp", "gambit-pure"] = "lp",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve strong Stackelberg equilibrium using a bilevel-to-LP reduction.

    The defender is the leader and commits to a mixed strategy x.
    For each attacker action j, solve the LP:

    max_x      U_D(:, j)^T x
    s.t.       x is a probability simplex
               U_A(:, j)^T x >= U_A(:, k)^T x for all k

    The best feasible j is selected (optimistic tie-breaking among follower best responses).

    method:
    - "lp": mixed-leader strong Stackelberg equilibrium via LP (default)
    - "gambit-pure": pure leader commitment with PyGambit extensive-form representation
    
    Args:
        defender_payoff: Shape (n_defender, n_attacker) payoff matrix
        attacker_payoff: Shape (n_defender, n_attacker) payoff matrix
        
    Returns:
        defender_sigma: Pure strategy as probability vector (one-hot)
        attacker_sigma: Pure best-response as probability vector (one-hot)
        defender_value: Defender's guaranteed payoff from Stackelberg commitment
    """
    d_payoff, a_payoff = _validate_bimatrix(defender_payoff, attacker_payoff)

    if method == "lp":
        return _solve_stackelberg_lp(d_payoff, a_payoff)
    if method == "gambit-pure":
        return _solve_stackelberg_pure_with_gambit_structure(d_payoff, a_payoff)
    raise ValueError(f"Unsupported stackelberg method: {method}")


def solve_stackelberg_security_game_gambit_pure(
    defender_payoff: np.ndarray,
    attacker_payoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convenience wrapper for the PyGambit pure-commitment Stackelberg path."""
    d_payoff, a_payoff = _validate_bimatrix(defender_payoff, attacker_payoff)
    return _solve_stackelberg_pure_with_gambit_structure(d_payoff, a_payoff)
