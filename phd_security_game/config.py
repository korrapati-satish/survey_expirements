from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    env_id: str = "CyberBattleChain-v0"
    chain_size: int = 10
    episodes_per_strategy: int = 20
    iterations_per_episode: int = 400
    seed: int = 42


@dataclass(frozen=True)
class EvolutionaryConfig:
    max_iterations: int = 20_000
    epsilon: float = 1e-6
    stable_window: int = 150


@dataclass(frozen=True)
class Ns3PenaltyConfig:
    latency_weight: float = 0.15
    packet_loss_weight: float = 2.0
    jitter_weight: float = 0.10
