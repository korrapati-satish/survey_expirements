from __future__ import annotations

import numpy as np
import pandas as pd

from phd_security_game.config import Ns3PenaltyConfig


_ALIASES: dict[str, tuple[str, ...]] = {
    "episode": ("episode", "run", "run_id", "simulation", "sim_id"),
    "step": ("step", "timestep", "time_step", "tick", "index"),
    "latency_ms": ("latency_ms", "delay_ms", "mean_delay_ms", "avg_delay_ms", "latency", "delay"),
    "jitter_ms": ("jitter_ms", "mean_jitter_ms", "avg_jitter_ms", "jitter", "jitter_delay_ms"),
    "packet_loss_rate": (
        "packet_loss_rate",
        "packet_loss",
        "loss_rate",
        "plr",
        "packet_loss_percent",
        "loss_percent",
    ),
}


def _find_first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    existing = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in existing:
            return existing[name]
    return None


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    rename_map: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        found = _find_first_present(out, aliases)
        if found and found != canonical:
            rename_map[found] = canonical

    if rename_map:
        out = out.rename(columns=rename_map)

    # Derive packet loss rate from count-style outputs if a direct rate is absent.
    if "packet_loss_rate" not in out.columns:
        if {"lost_packets", "tx_packets"}.issubset(set(out.columns)):
            tx = out["tx_packets"].astype(float)
            lost = out["lost_packets"].astype(float)
            out["packet_loss_rate"] = np.where(tx > 0.0, lost / tx, 0.0)
        elif {"lost_packets", "rx_packets"}.issubset(set(out.columns)):
            rx = out["rx_packets"].astype(float)
            lost = out["lost_packets"].astype(float)
            denom = rx + lost
            out["packet_loss_rate"] = np.where(denom > 0.0, lost / denom, 0.0)
        elif {"dropped_packets", "tx_packets"}.issubset(set(out.columns)):
            tx = out["tx_packets"].astype(float)
            dropped = out["dropped_packets"].astype(float)
            out["packet_loss_rate"] = np.where(tx > 0.0, dropped / tx, 0.0)

    return out


def _ensure_step_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "step" in out.columns:
        return out

    # Common ns-3 traces expose time rather than step; map ordered times to steps.
    time_col = _find_first_present(out, ("time_s", "time_ms", "time", "timestamp", "sim_time"))
    if time_col:
        out["step"] = out[time_col].rank(method="dense").astype(int)
    else:
        out["step"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _assert_required_ns3_columns(df: pd.DataFrame) -> None:
    required = {"episode", "step", "latency_ms", "packet_loss_rate", "jitter_ms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"NS-3 dataset is missing required columns: {missing}")


def load_ns3_trace_data(csv_path: str) -> pd.DataFrame:
    ns3_df = pd.read_csv(csv_path)
    ns3_df = _normalize_column_names(ns3_df)
    ns3_df = _ensure_step_column(ns3_df)

    # If episode is absent, apply the same ns-3 timeline across all episodes.
    if "episode" not in ns3_df.columns:
        ns3_df["episode"] = 0

    _assert_required_ns3_columns(ns3_df)

    out = ns3_df.copy()
    out["episode"] = out["episode"].astype(int)
    out["step"] = out["step"].astype(int)
    out["latency_ms"] = out["latency_ms"].astype(float)
    out["packet_loss_rate"] = out["packet_loss_rate"].astype(float)
    out["jitter_ms"] = out["jitter_ms"].astype(float)

    # Accept percentage-style rates (e.g., 2.5 means 2.5%).
    if float(out["packet_loss_rate"].max()) > 1.0:
        out["packet_loss_rate"] = out["packet_loss_rate"] / 100.0

    out["packet_loss_rate"] = out["packet_loss_rate"].clip(lower=0.0, upper=1.0)
    return out


def apply_ns3_network_conditions(
    traces_df: pd.DataFrame,
    ns3_df: pd.DataFrame,
    penalty_cfg: Ns3PenaltyConfig,
) -> pd.DataFrame:
    join_keys = ["episode", "step"]
    if int(ns3_df["episode"].nunique()) == 1 and int(traces_df["episode"].nunique()) > 1:
        # One ns-3 trajectory applied to all CyberBattleSim episodes.
        join_keys = ["step"]

    merged = traces_df.merge(ns3_df, on=join_keys, how="left")
    merged["latency_ms"] = merged["latency_ms"].fillna(float(ns3_df["latency_ms"].mean()))
    merged["packet_loss_rate"] = merged["packet_loss_rate"].fillna(float(ns3_df["packet_loss_rate"].mean()))
    merged["jitter_ms"] = merged["jitter_ms"].fillna(float(ns3_df["jitter_ms"].mean()))

    penalty = (
        penalty_cfg.latency_weight * (merged["latency_ms"] / 100.0)
        + penalty_cfg.packet_loss_weight * merged["packet_loss_rate"]
        + penalty_cfg.jitter_weight * (merged["jitter_ms"] / 50.0)
    )

    adjusted = merged.copy()
    adjusted["defender_utility"] = adjusted["defender_utility"] - penalty
    adjusted["attacker_utility"] = adjusted["attacker_utility"] + penalty
    return adjusted
