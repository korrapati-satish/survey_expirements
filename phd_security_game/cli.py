from __future__ import annotations

import argparse

from phd_security_game.config import EvolutionaryConfig, Ns3PenaltyConfig, SimulationConfig
from phd_security_game.pipeline.experiment_runner import run_full_experiment


def _print_strategy(label: str, names, probs) -> None:
    print(f"\n{label}")
    for name, p in zip(names, probs):
        print(f"  {name:35s}: {p:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CyberBattleSim + Gambit + OpenSpiel + NS-3 security-game experiment"
    )
    parser.add_argument("--env-id", type=str, default="CyberBattleChain-v0")
    parser.add_argument("--chain-size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--evo-max-iterations", type=int, default=20_000)
    parser.add_argument("--evo-epsilon", type=float, default=1e-6)
    parser.add_argument("--evo-stable-window", type=int, default=150)

    parser.add_argument("--enable-gambit-checks", action="store_true")
    parser.add_argument("--enable-openspiel-eval", action="store_true")
    parser.add_argument("--openspiel-eval-episodes", type=int, default=5000)

    parser.add_argument(
        "--ns3-csv",
        type=str,
        default=None,
        help=(
            "Path to NS-3 CSV. Supports canonical columns "
            "(episode, step, latency_ms, packet_loss_rate, jitter_ms) "
            "and common aliases like delay_ms/jitter/loss_rate."
        ),
    )
    parser.add_argument("--ns3-latency-weight", type=float, default=0.15)
    parser.add_argument("--ns3-packetloss-weight", type=float, default=2.0)
    parser.add_argument("--ns3-jitter-weight", type=float, default=0.10)

    parser.add_argument("--export-traces-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sim_cfg = SimulationConfig(
        env_id=args.env_id,
        chain_size=args.chain_size,
        episodes_per_strategy=args.episodes,
        iterations_per_episode=args.iterations,
        seed=args.seed,
    )
    evo_cfg = EvolutionaryConfig(
        max_iterations=args.evo_max_iterations,
        epsilon=args.evo_epsilon,
        stable_window=args.evo_stable_window,
    )
    ns3_cfg = Ns3PenaltyConfig(
        latency_weight=args.ns3_latency_weight,
        packet_loss_weight=args.ns3_packetloss_weight,
        jitter_weight=args.ns3_jitter_weight,
    )

    result = run_full_experiment(
        sim_cfg=sim_cfg,
        evo_cfg=evo_cfg,
        ns3_csv=args.ns3_csv,
        ns3_penalty_cfg=ns3_cfg,
        enable_gambit_checks=args.enable_gambit_checks,
        enable_openspiel_eval=args.enable_openspiel_eval,
        openspiel_eval_episodes=args.openspiel_eval_episodes,
        export_traces_csv=args.export_traces_csv,
    )

    print("Security Game Experiment Results")
    print("=" * 36)
    print(result.to_frame().to_string(index=False))

    # Strategy names are stable by construction of default_defender_strategies in runner.
    def_names = ["Adaptive IDS", "Resource-Constrained Monitoring", "Strict IDS"]
    att_names = ["connect", "local_vulnerability", "remote_vulnerability", "unknown"]

    _print_strategy("Defender Stackelberg Strategy", def_names[: len(result.defender_stackelberg_strategy)], result.defender_stackelberg_strategy)
    _print_strategy("Attacker Stackelberg Strategy", att_names[: len(result.attacker_stackelberg_strategy)], result.attacker_stackelberg_strategy)
    _print_strategy("Defender Evolutionary Strategy", def_names[: len(result.defender_evolutionary_strategy)], result.defender_evolutionary_strategy)
    _print_strategy("Attacker Evolutionary Strategy", att_names[: len(result.attacker_evolutionary_strategy)], result.attacker_evolutionary_strategy)


if __name__ == "__main__":
    main()
