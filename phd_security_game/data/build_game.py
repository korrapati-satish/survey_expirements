from __future__ import annotations

import numpy as np
import pandas as pd

from phd_security_game.game_types import SecurityGameConfig


def _assert_required_battle_columns(df: pd.DataFrame) -> None:
    required = {
        "defender_strategy",
        "attacker_action",
        "is_attack",
        "predicted_attack",
        "defender_utility",
        "attacker_utility",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Battle dataset is missing required columns: {missing}")


def _fill_sparse_matrix_cells(matrix: np.ndarray, fallback_value: float) -> np.ndarray:
    out = matrix.copy()
    mask = np.isnan(out)
    if np.any(mask):
        out[mask] = fallback_value
    return out


def _build_game_with_fixed_spaces(
    df: pd.DataFrame,
    defender_strategies: tuple[str, ...],
    attacker_actions: tuple[str, ...],
    global_fallback: pd.DataFrame,
) -> SecurityGameConfig:
    d_map = {name: i for i, name in enumerate(defender_strategies)}
    a_map = {name: j for j, name in enumerate(attacker_actions)}

    d_count = len(defender_strategies)
    a_count = len(attacker_actions)

    defender_payoff = np.full((d_count, a_count), np.nan, dtype=float)
    attacker_payoff = np.full((d_count, a_count), np.nan, dtype=float)
    detection_probability = np.full((d_count, a_count), np.nan, dtype=float)

    grouped = df.groupby(["defender_strategy", "attacker_action"], sort=False)
    for (d_name, a_name), g in grouped:
        i = d_map[str(d_name)]
        j = a_map[str(a_name)]
        defender_payoff[i, j] = float(g["defender_utility"].mean())
        attacker_payoff[i, j] = float(g["attacker_utility"].mean())
        detection_probability[i, j] = float(g["predicted_attack"].mean())

    defender_payoff = _fill_sparse_matrix_cells(defender_payoff, float(global_fallback["defender_utility"].mean()))
    attacker_payoff = _fill_sparse_matrix_cells(attacker_payoff, float(global_fallback["attacker_utility"].mean()))
    detection_probability = _fill_sparse_matrix_cells(detection_probability, float(global_fallback["predicted_attack"].mean()))

    attack_truth = (
        global_fallback.groupby("attacker_action", sort=False)["is_attack"]
        .mean()
        .reindex(attacker_actions)
        .to_numpy()
    )
    is_attack_action = tuple(attack_truth >= 0.5)

    cfg = SecurityGameConfig(
        defender_strategies=defender_strategies,
        attacker_actions=attacker_actions,
        is_attack_action=is_attack_action,
        defender_payoff=defender_payoff,
        attacker_payoff=attacker_payoff,
        detection_probability=detection_probability,
    )
    cfg.validate()
    return cfg


def build_game_from_battle_data(df: pd.DataFrame) -> SecurityGameConfig:
    _assert_required_battle_columns(df)

    defender_strategies = tuple(sorted(df["defender_strategy"].astype(str).unique().tolist()))
    attacker_actions = tuple(sorted(df["attacker_action"].astype(str).unique().tolist()))

    d_map = {name: i for i, name in enumerate(defender_strategies)}
    a_map = {name: j for j, name in enumerate(attacker_actions)}

    d_count = len(defender_strategies)
    a_count = len(attacker_actions)

    defender_payoff = np.full((d_count, a_count), np.nan, dtype=float)
    attacker_payoff = np.full((d_count, a_count), np.nan, dtype=float)
    detection_probability = np.full((d_count, a_count), np.nan, dtype=float)

    grouped = df.groupby(["defender_strategy", "attacker_action"], sort=False)
    for (d_name, a_name), g in grouped:
        i = d_map[str(d_name)]
        j = a_map[str(a_name)]
        defender_payoff[i, j] = float(g["defender_utility"].mean())
        attacker_payoff[i, j] = float(g["attacker_utility"].mean())
        detection_probability[i, j] = float(g["predicted_attack"].mean())

    defender_payoff = _fill_sparse_matrix_cells(defender_payoff, float(df["defender_utility"].mean()))
    attacker_payoff = _fill_sparse_matrix_cells(attacker_payoff, float(df["attacker_utility"].mean()))
    detection_probability = _fill_sparse_matrix_cells(detection_probability, float(df["predicted_attack"].mean()))

    attack_truth = df.groupby("attacker_action", sort=False)["is_attack"].mean().reindex(attacker_actions).to_numpy()
    is_attack_action = tuple(attack_truth >= 0.5)

    cfg = SecurityGameConfig(
        defender_strategies=defender_strategies,
        attacker_actions=attacker_actions,
        is_attack_action=is_attack_action,
        defender_payoff=defender_payoff,
        attacker_payoff=attacker_payoff,
        detection_probability=detection_probability,
    )
    cfg.validate()
    return cfg


def build_games_by_episode(df: pd.DataFrame) -> list[SecurityGameConfig]:
    """Build one empirical game per episode, using global spaces for consistency."""
    _assert_required_battle_columns(df)

    defender_strategies = tuple(sorted(df["defender_strategy"].astype(str).unique().tolist()))
    attacker_actions = tuple(sorted(df["attacker_action"].astype(str).unique().tolist()))

    out: list[SecurityGameConfig] = []
    for _, g in df.groupby("episode", sort=True):
        out.append(
            _build_game_with_fixed_spaces(
                df=g,
                defender_strategies=defender_strategies,
                attacker_actions=attacker_actions,
                global_fallback=df,
            )
        )

    if not out:
        raise ValueError("No episode data found to build per-episode games")
    return out
