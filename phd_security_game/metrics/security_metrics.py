from __future__ import annotations

import numpy as np

from phd_security_game.game_types import SecurityGameConfig


def expected_payoff(
    defender_strategy: np.ndarray,
    attacker_strategy: np.ndarray,
    defender_payoff: np.ndarray,
) -> float:
    return float(defender_strategy @ defender_payoff @ attacker_strategy)


def analytical_detection_metrics(
    config: SecurityGameConfig,
    defender_strategy: np.ndarray,
    attacker_strategy: np.ndarray,
    benign_reference_strategy: np.ndarray | None = None,
) -> tuple[float, float]:
    is_attack = np.asarray(config.is_attack_action, dtype=bool)
    is_benign = ~is_attack

    predicted_attack_given_action = defender_strategy @ config.detection_probability

    benign_strategy = attacker_strategy if benign_reference_strategy is None else benign_reference_strategy

    attack_mass = float(attacker_strategy[is_attack].sum())
    benign_mass = float(benign_strategy[is_benign].sum())

    if attack_mass > 0.0:
        adr = float(np.dot(attacker_strategy[is_attack], predicted_attack_given_action[is_attack]) / attack_mass)
    else:
        adr = 0.0

    if benign_mass > 0.0:
        fpr = float(np.dot(benign_strategy[is_benign], predicted_attack_given_action[is_benign]) / benign_mass)
    else:
        fpr = 0.0

    return float(adr * 100.0), float(fpr * 100.0)
