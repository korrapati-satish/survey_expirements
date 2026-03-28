from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from phd_security_game.config import EvolutionaryConfig, Ns3PenaltyConfig, SimulationConfig
from phd_security_game.data.build_game import build_game_from_battle_data, build_games_by_episode
from phd_security_game.metrics.security_metrics import analytical_detection_metrics, expected_payoff
from phd_security_game.models.evolutionary import run_evolutionary_dynamics
from phd_security_game.models.baseline_gambit import solve_gambit_nash_baseline
from phd_security_game.models.baseline_static import solve_static_defender_baseline
from phd_security_game.models.gambit_solver import (
    count_extensive_form_equilibria,
    count_normal_form_equilibria,
)
from phd_security_game.models.openspiel_eval import evaluate_with_openspiel_rl
from phd_security_game.models.stackelberg import solve_stackelberg_security_game
from phd_security_game.simulators.cyberbattlesim_runner import (
    collect_microsoft_battlesim_data,
    default_defender_strategies,
)
from phd_security_game.simulators.ns3 import apply_ns3_network_conditions, load_ns3_trace_data
from phd_security_game.game_types import ExperimentResult


def _empirical_action_mix(traces_df: pd.DataFrame, attacker_actions: tuple[str, ...]) -> np.ndarray:
    counts = (
        traces_df.groupby("attacker_action", sort=False)
        .size()
        .reindex(attacker_actions)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(len(attacker_actions), 1.0 / len(attacker_actions), dtype=float)
    return counts / total


def _normalize_simplex(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = float(clipped.sum())
    if s <= 0.0:
        return np.full(clipped.size, 1.0 / clipped.size, dtype=float)
    return clipped / s


def run_full_experiment(
    sim_cfg: SimulationConfig,
    evo_cfg: EvolutionaryConfig,
    ns3_csv: Optional[str] = None,
    ns3_penalty_cfg: Ns3PenaltyConfig = Ns3PenaltyConfig(),
    enable_gambit_checks: bool = False,
    enable_openspiel_eval: bool = False,
    openspiel_eval_episodes: int = 5000,
    export_traces_csv: Optional[str] = None,
) -> ExperimentResult:
    print("\n" + "="*80)
    print("STAGE 1: Collecting CyberBattleSim Data")
    print("="*80)
    traces_df = collect_microsoft_battlesim_data(default_defender_strategies(), sim_cfg)
    
    print(f"\n✓ Simulation data collected")
    print(f"  - Total traces: {len(traces_df)}")
    print(f"  - Episodes: {traces_df['episode'].max() + 1}")
    print(f"  - Columns: {list(traces_df.columns)}")
    print(f"  - Defender strategies: {traces_df['defender_strategy'].unique().tolist()}")
    print(f"  - Unique attacker actions: {traces_df['attacker_action'].nunique()}")

    if ns3_csv:
        print("\n" + "="*80)
        print("STAGE 2: Applying NS-3 Network Conditions")
        print("="*80)
        ns3_df = load_ns3_trace_data(ns3_csv)
        print(f"\n✓ NS-3 trace data loaded ({len(ns3_df)} rows)")
        traces_df = apply_ns3_network_conditions(traces_df, ns3_df, ns3_penalty_cfg)
        print(f"✓ Network penalties applied")
        print(f"  - Penalty config: {ns3_penalty_cfg}")
    else:
        print("\n" + "="*80)
        print("STAGE 2: NS-3 Network Conditions")
        print("="*80)
        print("✓ Skipped (no NS-3 CSV provided)")

    if export_traces_csv:
        traces_df.to_csv(export_traces_csv, index=False)

    print("\n" + "="*80)
    print("STAGE 3: Building Empirical Games")
    print("="*80)
    game_cfg = build_game_from_battle_data(traces_df)
    episode_cfgs = build_games_by_episode(traces_df)
    empirical_mix = _empirical_action_mix(traces_df, tuple(game_cfg.attacker_actions))
    
    print(f"\n✓ Games constructed")
    print(f"  - Global game: {len(game_cfg.defender_strategies)} defender × {len(game_cfg.attacker_actions)} attacker actions")
    print(f"  - Per-episode games: {len(episode_cfgs)}")
    print(f"  - Defender strategies: {list(game_cfg.defender_strategies)}")
    print(f"  - Attacker actions (sample): {list(game_cfg.attacker_actions[:5])} {'...' if len(game_cfg.attacker_actions) > 5 else ''}")
    print(f"  - Empirical action mix (top 3): {sorted(zip(list(game_cfg.attacker_actions), empirical_mix), key=lambda x: x[1], reverse=True)[:3]}")

    stackelberg_payoffs: list[float] = []
    evolutionary_payoffs: list[float] = []
    stackelberg_adrs: list[float] = []
    stackelberg_fprs: list[float] = []
    evolutionary_adrs: list[float] = []
    evolutionary_fprs: list[float] = []
    baseline_payoffs: list[float] = []
    baseline_adrs: list[float] = []
    baseline_fprs: list[float] = []
    static_payoffs: list[float] = []
    static_adrs: list[float] = []
    static_fprs: list[float] = []
    convergence_times: list[float] = []
    convergence_steps: list[int] = []

    defender_stack_profiles: list[np.ndarray] = []
    attacker_stack_profiles: list[np.ndarray] = []
    defender_evo_profiles: list[np.ndarray] = []
    attacker_evo_profiles: list[np.ndarray] = []

    print("\n" + "="*80)
    print("STAGE 4: Solving Per-Episode Games")
    print("="*80)

    static_attacker = _normalize_simplex(empirical_mix)
    defender_static, _ = solve_static_defender_baseline(
        game_cfg.defender_payoff,
        static_attacker,
    )

    for ep_idx, ep_cfg in enumerate(episode_cfgs):
        attack_mask = np.asarray(ep_cfg.is_attack_action, dtype=bool)
        attack_indices = np.flatnonzero(attack_mask)
        if attack_indices.size == 0:
            continue

        defender_payoff_attack = ep_cfg.defender_payoff[:, attack_indices]
        attacker_payoff_attack = ep_cfg.attacker_payoff[:, attack_indices]

        defender_stack, attacker_stack_attack, _ = solve_stackelberg_security_game(
            defender_payoff_attack,
            attacker_payoff_attack,
        )

        attacker_stack = np.zeros(len(ep_cfg.attacker_actions), dtype=float)
        attacker_stack[attack_indices] = attacker_stack_attack

        defender_evo_attack, attacker_evo_attack, convergence_time, convergence_iters = run_evolutionary_dynamics(
            defender_payoff=defender_payoff_attack,
            attacker_payoff=attacker_payoff_attack,
            cfg=evo_cfg,
        )

        defender_base_attack, attacker_base_attack, _ = solve_gambit_nash_baseline(
            defender_payoff_attack,
            attacker_payoff_attack,
        )

        attacker_evo = np.zeros(len(ep_cfg.attacker_actions), dtype=float)
        attacker_evo[attack_indices] = attacker_evo_attack
        defender_evo = defender_evo_attack

        attacker_base = np.zeros(len(ep_cfg.attacker_actions), dtype=float)
        attacker_base[attack_indices] = attacker_base_attack
        defender_base = defender_base_attack

        stack_pay = expected_payoff(
            defender_strategy=defender_stack,
            attacker_strategy=attacker_stack,
            defender_payoff=ep_cfg.defender_payoff,
        )
        evo_pay = expected_payoff(
            defender_strategy=defender_evo,
            attacker_strategy=attacker_evo,
            defender_payoff=ep_cfg.defender_payoff,
        )
        base_pay = expected_payoff(
            defender_strategy=defender_base,
            attacker_strategy=attacker_base,
            defender_payoff=ep_cfg.defender_payoff,
        )
        static_pay = expected_payoff(
            defender_strategy=defender_static,
            attacker_strategy=static_attacker,
            defender_payoff=ep_cfg.defender_payoff,
        )

        stack_adr, stack_fpr = analytical_detection_metrics(
            config=ep_cfg,
            defender_strategy=defender_stack,
            attacker_strategy=attacker_stack,
            benign_reference_strategy=empirical_mix,
        )
        evo_adr, evo_fpr = analytical_detection_metrics(
            config=ep_cfg,
            defender_strategy=defender_evo,
            attacker_strategy=attacker_evo,
            benign_reference_strategy=empirical_mix,
        )
        base_adr, base_fpr = analytical_detection_metrics(
            config=ep_cfg,
            defender_strategy=defender_base,
            attacker_strategy=attacker_base,
            benign_reference_strategy=empirical_mix,
        )
        static_adr, static_fpr = analytical_detection_metrics(
            config=ep_cfg,
            defender_strategy=defender_static,
            attacker_strategy=static_attacker,
            benign_reference_strategy=empirical_mix,
        )

        stackelberg_payoffs.append(float(stack_pay))
        evolutionary_payoffs.append(float(evo_pay))
        stackelberg_adrs.append(float(stack_adr))
        stackelberg_fprs.append(float(stack_fpr))
        evolutionary_adrs.append(float(evo_adr))
        evolutionary_fprs.append(float(evo_fpr))
        baseline_payoffs.append(float(base_pay))
        baseline_adrs.append(float(base_adr))
        baseline_fprs.append(float(base_fpr))
        static_payoffs.append(float(static_pay))
        static_adrs.append(float(static_adr))
        static_fprs.append(float(static_fpr))
        convergence_times.append(float(convergence_time))
        convergence_steps.append(int(convergence_iters))
        defender_stack_profiles.append(defender_stack)
        attacker_stack_profiles.append(attacker_stack)
        defender_evo_profiles.append(defender_evo)
        attacker_evo_profiles.append(attacker_evo)
        
        if (ep_idx + 1) % 5 == 0 or ep_idx == 0 or ep_idx == len(episode_cfgs) - 1:
            print(f"\n  Episode {ep_idx + 1}/{len(episode_cfgs)}:")
            print(f"    Stackelberg: Payoff={stack_pay:.4f}, ADR={stack_adr:.4f}, FPR={stack_fpr:.4f}")
            print(f"    Evolutionary: Payoff={evo_pay:.4f}, ADR={evo_adr:.4f}, FPR={evo_fpr:.4f}, Conv.Time={convergence_time:.2f}s")
            print(f"    Baseline (Gambit Nash): Payoff={base_pay:.4f}, ADR={base_adr:.4f}, FPR={base_fpr:.4f}")
            print(f"    Baseline (Static Fixed): Payoff={static_pay:.4f}, ADR={static_adr:.4f}, FPR={static_fpr:.4f}")

    if not stackelberg_payoffs:
        raise ValueError("No valid per-episode games could be solved")

    print("\n" + "="*80)
    print("STAGE 5: Aggregating Results")
    print("="*80)
    
    defender_stack = _normalize_simplex(np.mean(np.vstack(defender_stack_profiles), axis=0))
    attacker_stack = _normalize_simplex(np.mean(np.vstack(attacker_stack_profiles), axis=0))
    defender_evo = _normalize_simplex(np.mean(np.vstack(defender_evo_profiles), axis=0))
    attacker_evo = _normalize_simplex(np.mean(np.vstack(attacker_evo_profiles), axis=0))

    stackelberg_payoff_gain = float(np.mean(stackelberg_payoffs))
    evolutionary_payoff_gain = float(np.mean(evolutionary_payoffs))
    stackelberg_adr = float(np.mean(stackelberg_adrs))
    stackelberg_fpr = float(np.mean(stackelberg_fprs))
    evolutionary_adr = float(np.mean(evolutionary_adrs))
    evolutionary_fpr = float(np.mean(evolutionary_fprs))
    baseline_payoff_gain = float(np.mean(baseline_payoffs))
    baseline_adr = float(np.mean(baseline_adrs))
    baseline_fpr = float(np.mean(baseline_fprs))
    static_payoff_gain = float(np.mean(static_payoffs))
    static_adr = float(np.mean(static_adrs))
    static_fpr = float(np.mean(static_fprs))
    convergence_time = float(np.mean(convergence_times))
    convergence_iters = int(round(float(np.mean(convergence_steps))))

    print(f"\n✓ Aggregated across {len(stackelberg_payoffs)} episodes:")
    print(f"  Stackelberg:")
    print(f"    - Avg Payoff: {stackelberg_payoff_gain:.6f}")
    print(f"    - Avg ADR: {stackelberg_adr:.6f}")
    print(f"    - Avg FPR: {stackelberg_fpr:.6f}")
    print(f"  Evolutionary:")
    print(f"    - Avg Payoff: {evolutionary_payoff_gain:.6f}")
    print(f"    - Avg ADR: {evolutionary_adr:.6f}")
    print(f"    - Avg FPR: {evolutionary_fpr:.6f}")
    print(f"    - Avg Convergence Time: {convergence_time:.2f}s")
    print(f"    - Avg Convergence Iterations: {convergence_iters}")
    print(f"  Baseline (Gambit Nash):")
    print(f"    - Avg Payoff: {baseline_payoff_gain:.6f}")
    print(f"    - Avg ADR: {baseline_adr:.6f}")
    print(f"    - Avg FPR: {baseline_fpr:.6f}")
    print(f"  Baseline (Static Fixed):")
    print(f"    - Avg Payoff: {static_payoff_gain:.6f}")
    print(f"    - Avg ADR: {static_adr:.6f}")
    print(f"    - Avg FPR: {static_fpr:.6f}")

    gambit_normal_count = 0
    gambit_extensive_count = 0
    if enable_gambit_checks:
        print("\n" + "="*80)
        print("STAGE 6: Gambit Equilibrium Checks")
        print("="*80)
        gambit_normal_count = count_normal_form_equilibria(game_cfg.defender_payoff, game_cfg.attacker_payoff)
        gambit_extensive_count = count_extensive_form_equilibria(game_cfg.defender_payoff, game_cfg.attacker_payoff)
        print(f"\n✓ Equilibria counted (global game):")
        print(f"  - Normal Form Equilibria: {gambit_normal_count}")
        print(f"  - Extensive Form Equilibria: {gambit_extensive_count}")
    else:
        print("\n" + "="*80)
        print("STAGE 6: Gambit Equilibrium Checks")
        print("="*80)
        print("✓ Skipped (--enable-gambit-checks not set)")

    openspiel_estimate_stackelberg = None
    openspiel_estimate_evolutionary = None
    if enable_openspiel_eval:
        print("\n" + "="*80)
        print("STAGE 7: OpenSpiel RL Evaluation")
        print("="*80)
        openspiel_estimate_stackelberg = evaluate_with_openspiel_rl(
            defender_payoff=game_cfg.defender_payoff,
            attacker_strategy=attacker_stack,
            episodes=openspiel_eval_episodes,
            seed=sim_cfg.seed,
        )
        openspiel_estimate_evolutionary = evaluate_with_openspiel_rl(
            defender_payoff=game_cfg.defender_payoff,
            attacker_strategy=attacker_evo,
            episodes=openspiel_eval_episodes,
            seed=sim_cfg.seed,
        )
        print(f"\n✓ RL evaluation completed ({openspiel_eval_episodes} episodes):")
        print(f"  - Stackelberg payoff estimate: {openspiel_estimate_stackelberg:.6f}")
        print(f"  - Evolutionary payoff estimate: {openspiel_estimate_evolutionary:.6f}")
    else:
        print("\n" + "="*80)
        print("STAGE 7: OpenSpiel RL Evaluation")
        print("="*80)
        print("✓ Skipped (--enable-openspiel-eval not set)")

    payoff_delta = float(evolutionary_payoff_gain - stackelberg_payoff_gain)
    adr_delta = float(evolutionary_adr - stackelberg_adr)
    fpr_delta = float(evolutionary_fpr - stackelberg_fpr)

    print("\n" + "="*80)
    print("STAGE 8: Final Summary")
    print("="*80)
    print(f"\n✓ Experiment completed successfully!")
    print(f"\n  Payoff Comparison:")
    print(f"    - Stackelberg: {stackelberg_payoff_gain:.6f}")
    print(f"    - Evolutionary: {evolutionary_payoff_gain:.6f}")
    print(f"    - Delta (Evo - Stack): {payoff_delta:.6f}")
    print(f"\n  Detection Rate Comparison:")
    print(f"    - Stackelberg ADR: {stackelberg_adr:.6f}")
    print(f"    - Evolutionary ADR: {evolutionary_adr:.6f}")
    print(f"    - Delta (Evo - Stack): {adr_delta:.6f}")
    print(f"\n  False Positive Rate Comparison:")
    print(f"    - Stackelberg FPR: {stackelberg_fpr:.6f}")
    print(f"    - Evolutionary FPR: {evolutionary_fpr:.6f}")
    print(f"    - Delta (Evo - Stack): {fpr_delta:.6f}")
    print("\n" + "="*80 + "\n")

    return ExperimentResult(
        defender_stackelberg_strategy=defender_stack,
        attacker_stackelberg_strategy=attacker_stack,
        defender_evolutionary_strategy=defender_evo,
        attacker_evolutionary_strategy=attacker_evo,
        payoff_gain=float(stackelberg_payoff_gain),
        attack_detection_rate=stackelberg_adr,
        false_positive_rate=stackelberg_fpr,
        stackelberg_payoff_gain=float(stackelberg_payoff_gain),
        evolutionary_payoff_gain=float(evolutionary_payoff_gain),
        payoff_gain_delta_evo_minus_stack=payoff_delta,
        stackelberg_attack_detection_rate=stackelberg_adr,
        evolutionary_attack_detection_rate=evolutionary_adr,
        adr_delta_evo_minus_stack=adr_delta,
        stackelberg_false_positive_rate=stackelberg_fpr,
        evolutionary_false_positive_rate=evolutionary_fpr,
        fpr_delta_evo_minus_stack=fpr_delta,
        convergence_time_seconds=convergence_time,
        convergence_iterations=convergence_iters,
        gambit_normal_form_equilibria=gambit_normal_count,
        gambit_extensive_form_equilibria=gambit_extensive_count,
        openspiel_rl_payoff_estimate=openspiel_estimate_evolutionary,
        openspiel_rl_payoff_estimate_stackelberg=openspiel_estimate_stackelberg,
        openspiel_rl_payoff_estimate_evolutionary=openspiel_estimate_evolutionary,
        baseline_payoff_gain=baseline_payoff_gain,
        baseline_attack_detection_rate=baseline_adr,
        baseline_false_positive_rate=baseline_fpr,
        static_baseline_payoff_gain=static_payoff_gain,
        static_baseline_attack_detection_rate=static_adr,
        static_baseline_false_positive_rate=static_fpr,
    )
