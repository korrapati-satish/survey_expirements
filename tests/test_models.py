from __future__ import annotations

import numpy as np

from phd_security_game.config import EvolutionaryConfig
from phd_security_game.models.evolutionary import run_evolutionary_attacker_dynamics
from phd_security_game.models.baseline_gambit import solve_gambit_nash_baseline
from phd_security_game.models.baseline_static import solve_static_defender_baseline
from phd_security_game.models.stackelberg import (
    solve_stackelberg_security_game,
    solve_stackelberg_security_game_gambit_pure,
)


def test_stackelberg_returns_valid_profiles() -> None:
    d_payoff = np.array([[5.0, 1.0], [2.0, 3.0]], dtype=float)
    a_payoff = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=float)

    d_sigma, a_sigma, value = solve_stackelberg_security_game(d_payoff, a_payoff)

    assert np.isclose(d_sigma.sum(), 1.0)
    assert np.isclose(a_sigma.sum(), 1.0)
    assert np.all(d_sigma >= 0.0)
    assert np.all(a_sigma >= 0.0)
    assert np.isfinite(value)


def test_stackelberg_gambit_pure_returns_valid_profiles() -> None:
    d_payoff = np.array([[5.0, 1.0], [2.0, 3.0]], dtype=float)
    a_payoff = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=float)

    d_sigma, a_sigma, value = solve_stackelberg_security_game_gambit_pure(d_payoff, a_payoff)

    assert np.isclose(d_sigma.sum(), 1.0)
    assert np.isclose(a_sigma.sum(), 1.0)
    assert np.all(d_sigma >= 0.0)
    assert np.all(a_sigma >= 0.0)
    assert np.isfinite(value)


def test_stackelberg_method_switch_works() -> None:
    d_payoff = np.array([[4.0, 0.0, 1.0], [2.0, 3.0, 2.0]], dtype=float)
    a_payoff = np.array([[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]], dtype=float)

    d_lp, a_lp, v_lp = solve_stackelberg_security_game(d_payoff, a_payoff, method="lp")
    d_gp, a_gp, v_gp = solve_stackelberg_security_game(d_payoff, a_payoff, method="gambit-pure")

    assert np.isclose(d_lp.sum(), 1.0)
    assert np.isclose(a_lp.sum(), 1.0)
    assert np.isfinite(v_lp)
    assert np.isclose(d_gp.sum(), 1.0)
    assert np.isclose(a_gp.sum(), 1.0)
    assert np.isfinite(v_gp)


def test_baseline_gambit_nash_returns_valid_profiles() -> None:
    d_payoff = np.array([[4.0, 0.0, 1.0], [2.0, 3.0, 2.0]], dtype=float)
    a_payoff = np.array([[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]], dtype=float)

    d_sigma, a_sigma, value = solve_gambit_nash_baseline(d_payoff, a_payoff)

    assert np.isclose(d_sigma.sum(), 1.0)
    assert np.isclose(a_sigma.sum(), 1.0)
    assert np.all(d_sigma >= 0.0)
    assert np.all(a_sigma >= 0.0)
    assert np.isfinite(value)


def test_baseline_static_returns_valid_profile() -> None:
    d_payoff = np.array([[4.0, 1.0, 2.0], [2.0, 3.0, 2.5]], dtype=float)
    attacker_ref = np.array([0.5, 0.25, 0.25], dtype=float)

    d_sigma, value = solve_static_defender_baseline(d_payoff, attacker_ref)

    assert np.isclose(d_sigma.sum(), 1.0)
    assert np.all(d_sigma >= 0.0)
    assert np.count_nonzero(d_sigma > 1e-12) == 1
    assert np.isfinite(value)


def test_evolutionary_dynamics_probability_simplex() -> None:
    defender = np.array([1.0, 0.0], dtype=float)
    defender_payoff = np.array([[1.0, 0.5, 0.25], [0.2, 0.7, 0.6]], dtype=float)
    attacker_payoff = np.array([[2.0, 1.0, 0.5], [0.0, 0.5, 1.0]], dtype=float)

    x, seconds, iters = run_evolutionary_attacker_dynamics(
        defender_strategy=defender,
        defender_payoff=defender_payoff,
        attacker_payoff=attacker_payoff,
        cfg=EvolutionaryConfig(max_iterations=500, stable_window=20),
    )

    assert np.isclose(x.sum(), 1.0)
    assert np.all(x >= 0.0)
    assert seconds >= 0.0
    assert iters > 0
