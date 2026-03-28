from __future__ import annotations

import numpy as np


def create_extensive_game_from_bimatrix(defender_payoff: np.ndarray, attacker_payoff: np.ndarray):
    import pygambit as gbt  # type: ignore

    m, n = defender_payoff.shape
    game = gbt.Game.new_tree(players=["Defender", "Attacker"], title="Stackelberg Extensive Security Game")

    defender_actions = [f"D{i}" for i in range(m)]
    attacker_actions = [f"A{j}" for j in range(n)]

    game.append_move(game.root, "Defender", defender_actions)
    game.append_move(game.root.children, "Attacker", attacker_actions)

    for i in range(m):
        for j in range(n):
            outcome = game.add_outcome([float(defender_payoff[i, j]), float(attacker_payoff[i, j])])
            game.set_outcome(game.root.children[i].children[j], outcome)

    return game


def count_normal_form_equilibria(defender_payoff: np.ndarray, attacker_payoff: np.ndarray) -> int:
    import pygambit as gbt  # type: ignore

    game = gbt.Game.from_arrays(defender_payoff, attacker_payoff)
    result = gbt.nash.enummixed_solve(game)
    return int(len(result.equilibria))


def count_extensive_form_equilibria(defender_payoff: np.ndarray, attacker_payoff: np.ndarray) -> int:
    import pygambit as gbt  # type: ignore

    game = create_extensive_game_from_bimatrix(defender_payoff, attacker_payoff)
    result = gbt.nash.enummixed_solve(game)
    return int(len(result.equilibria))
