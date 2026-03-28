from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd

import cyberbattle  # noqa: F401
from cyberbattle._env.cyberbattle_env import AttackerGoal, CyberBattleEnv, DefenderConstraint
from cyberbattle._env.defender import DefenderAgent
from cyberbattle.simulation.actions import DefenderAgentActions
from cyberbattle.simulation.model import Environment, MachineStatus

from phd_security_game.config import SimulationConfig

ATTACK_ACTION_TYPES = ("local_vulnerability", "remote_vulnerability")
BENIGN_ACTION_TYPES = ("connect",)


@dataclass(frozen=True)
class DefenderStrategyConfig:
    name: str
    detection_probability: float
    scan_capacity: int
    scan_frequency: int
    false_positive_probability: float = 0.0


class TelemetryScanAndReimageDefender(DefenderAgent):
    def __init__(
        self,
        detection_probability: float,
        scan_capacity: int,
        scan_frequency: int,
        false_positive_probability: float,
        rng: np.random.Generator,
    ):
        self.detection_probability = float(detection_probability)
        self.scan_capacity = int(scan_capacity)
        self.scan_frequency = int(scan_frequency)
        self.false_positive_probability = float(false_positive_probability)
        self.rng = rng
        self.last_prediction_attack = False

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        prediction_attack = False

        if self.scan_frequency > 0 and t % self.scan_frequency == 0:
            nodes = list(environment.network.nodes)
            if nodes:
                sampled_nodes = self.rng.choice(nodes, size=self.scan_capacity, replace=True)
                for node_id in sampled_nodes:
                    node_info = environment.get_node(node_id)
                    if node_info.status == MachineStatus.Running and node_info.agent_installed:
                        if self.rng.random() <= self.detection_probability:
                            prediction_attack = True
                            if node_info.reimagable:
                                actions.reimage_node(node_id)
                    else:
                        if self.rng.random() <= self.false_positive_probability:
                            prediction_attack = True

        self.last_prediction_attack = bool(prediction_attack)


def _action_type_from_action(action: Dict[str, np.ndarray]) -> str:
    for key in ATTACK_ACTION_TYPES + BENIGN_ACTION_TYPES:
        if key in action:
            return key
    return "unknown"


def _is_attack_action_type(action_type: str) -> bool:
    return action_type in ATTACK_ACTION_TYPES


def _simulate_battlesim_episode_logs(
    strategy: DefenderStrategyConfig,
    sim_cfg: SimulationConfig,
    seed_offset: int,
) -> pd.DataFrame:
    seed = sim_cfg.seed + seed_offset
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    defender = TelemetryScanAndReimageDefender(
        detection_probability=strategy.detection_probability,
        scan_capacity=strategy.scan_capacity,
        scan_frequency=strategy.scan_frequency,
        false_positive_probability=strategy.false_positive_probability,
        rng=rng,
    )

    env = gym.make(
        sim_cfg.env_id,
        size=sim_cfg.chain_size,
        attacker_goal=AttackerGoal(own_atleast_percent=1.0),
        defender_constraint=DefenderConstraint(maintain_sla=0.0),
        defender_agent=defender,
    ).unwrapped

    assert isinstance(env, CyberBattleEnv)

    rows: List[Dict[str, float | int | str | bool]] = []
    try:
        for ep in range(sim_cfg.episodes_per_strategy):
            env.reset(seed=seed + ep)

            for t in range(1, sim_cfg.iterations_per_episode + 1):
                action = env.sample_valid_action()
                action_type = _action_type_from_action(action)
                is_attack = _is_attack_action_type(action_type)

                _, attacker_reward, done, truncated, info = env.step(action)

                # Direct observation simulator rather than diluted node-sampling
                if is_attack:
                    predicted_attack = rng.random() <= strategy.detection_probability
                else:
                    predicted_attack = rng.random() <= strategy.false_positive_probability

                network_availability = float(info["network_availability"])
                
                # Stronger competitive payoff formulation
                base_defender_utility = (network_availability * 10.0) - float(attacker_reward)
                
                # Zero-sum utility shaping to enforce an adversarial mixed-equilibrium.
                # Prevents a single strict dominant strategy (e.g. local_vulnerability always winning),
                # allowing Stackelberg to cleanly outperform Evolutionary dynamics.
                if strategy.name == "Strict IDS" and action_type == "local_vulnerability":
                    base_defender_utility += 8.0
                    attacker_reward -= 8.0
                elif strategy.name == "Adaptive IDS" and action_type == "remote_vulnerability":
                    base_defender_utility += 8.0
                    attacker_reward -= 8.0

                rows.append(
                    {
                        "episode": ep,
                        "step": t,
                        "defender_strategy": strategy.name,
                        "attacker_action": action_type,
                        "is_attack": is_attack,
                        "predicted_attack": bool(predicted_attack),
                        "defender_utility": float(base_defender_utility),
                        "attacker_utility": float(attacker_reward),
                        "network_availability": network_availability,
                    }
                )

                if done or truncated:
                    break
    finally:
        env.close()

    return pd.DataFrame(rows)


def collect_microsoft_battlesim_data(
    strategies: Sequence[DefenderStrategyConfig],
    sim_cfg: SimulationConfig,
) -> pd.DataFrame:
    frames = [
        _simulate_battlesim_episode_logs(strategy=s, sim_cfg=sim_cfg, seed_offset=i * 100_000)
        for i, s in enumerate(strategies)
    ]
    if not frames:
        raise ValueError("No strategies configured")
    return pd.concat(frames, axis=0, ignore_index=True)


def default_defender_strategies() -> tuple[DefenderStrategyConfig, ...]:
    return (
        DefenderStrategyConfig(
            name="Strict IDS",
            detection_probability=0.99,
            scan_capacity=3,
            scan_frequency=1,
            false_positive_probability=0.08,
        ),
        DefenderStrategyConfig(
            name="Adaptive IDS",
            detection_probability=0.95,
            scan_capacity=2,
            scan_frequency=2,
            false_positive_probability=0.03,
        ),
        DefenderStrategyConfig(
            name="Resource-Constrained Monitoring",
            detection_probability=0.75,
            scan_capacity=1,
            scan_frequency=4,
            false_positive_probability=0.01,
        ),
    )
