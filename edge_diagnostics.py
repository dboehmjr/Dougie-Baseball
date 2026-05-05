"""
Diagnose whether model edge translates into betting edge.

Reads data/walkforward_results.csv and writes grouped ROI / hit-rate reports:
  - data/edge_diagnostics_summary.csv
  - data/edge_diagnostics_by_edge.csv
  - data/edge_diagnostics_by_odds.csv
  - data/edge_diagnostics_by_month.csv
  - data/edge_diagnostics_by_season_phase.csv
  - data/edge_diagnostics_by_side.csv
  - data/edge_diagnostics_by_favdog.csv
  - data/edge_diagnostics_by_sp_throws.csv
  - data/edge_diagnostics_by_opp_sp_throws.csv

If future odds files include closing prices/probabilities, CLV columns are
included automatically. The current Action Network caches only have one
consensus snapshot, so CLV will be unavailable for now.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_PATH = os.path.join(DATA_DIR, "walkforward_results.csv")

EDGE_BINS = [-np.inf, 0.00, 0.01, 0.02, 0.03, 0.05, 0.08, np.inf]
EDGE_LABELS = ["<=0%", "0-1%", "1-2%", "2-3%", "3-5%", "5-8%", "8%+"]

CLV_BINS = [-np.inf, -0.03, -0.01, 0.00, 0.01, 0.03, np.inf]
CLV_LABELS = ["<-3%", "-3:-1%", "-1:0%", "0:1%", "1:3%", "3%+"]

ODDS_BINS = [-np.inf, -180, -140, -115, 100, 130, 170, np.inf]
ODDS_LABELS = ["<-180", "-180:-140", "-140:-115", "-115:+100", "+100:+130", "+130:+170", "+170+"]


def _safe_roi(pnl: float, stake: float) -> float:
    return float(pnl / stake) if stake and stake > 0 else np.nan


def _summarize(group: pd.DataFrame) -> pd.Series:
    bets = group[group["stake"] > 0].copy()
    candidates = group[group["edge"].notna()].copy()
    stake = float(bets["stake"].sum())
    pnl = float(bets["game_pnl"].sum())

    out = {
        "candidate_games": len(candidates),
        "bets": len(bets),
        "bet_rate": len(bets) / len(candidates) if len(candidates) else np.nan,
        "wins": int(bets["won"].sum()) if len(bets) else 0,
        "win_rate": float(bets["won"].mean()) if len(bets) else np.nan,
        "avg_edge": float(bets["edge"].mean()) if len(bets) else np.nan,
        "avg_model_prob": float(bets["model_prob"].mean()) if len(bets) else np.nan,
        "avg_vegas_prob": float(bets["vegas_prob"].mean()) if len(bets) else np.nan,
        "avg_odds": float(bets["odds"].mean()) if len(bets) else np.nan,
        "staked": stake,
        "pnl": pnl,
        "roi": _safe_roi(pnl, stake),
    }

    if "clv_prob" in bets.columns:
        out["avg_clv_prob"] = float(bets["clv_prob"].mean())
    if "clv_price" in bets.columns:
        out["avg_clv_price"] = float(bets["clv_price"].mean())
    return pd.Series(out)


def _write_report(df: pd.DataFrame, by: str, filename: str) -> pd.DataFrame:
    report = (
        df.groupby(by, dropna=False)
        .apply(_summarize)
        .reset_index()
    )
    path = os.path.join(DATA_DIR, filename)
    report.to_csv(path, index=False)
    return report


def build_reports(path: str = RESULTS_PATH) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["edge"].notna()].copy()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    if "season_phase" not in df.columns:
        df["season_phase"] = np.select(
            [df["date"].dt.month <= 4, df["date"].dt.month >= 9],
            ["early", "late"],
            default="mid",
        )
    df["edge_bucket"] = pd.cut(df["edge"], EDGE_BINS, labels=EDGE_LABELS, right=False)
    df["odds_bucket"] = pd.cut(df["odds"], ODDS_BINS, labels=ODDS_LABELS, right=False)

    if "is_favorite" in df.columns:
        df["favdog"] = np.where(df["is_favorite"], "favorite", "underdog")
    else:
        df["favdog"] = np.where(df["odds"] < 0, "favorite", "underdog")

    if "candidate_side" not in df.columns:
        df["candidate_side"] = df["bet_side"]
    if "side_home_away" in df.columns:
        df["home_or_away"] = df["side_home_away"].fillna("unknown")
    else:
        df["home_or_away"] = np.where(
            df["candidate_side"].eq(df.get("home_team")),
            "home",
            np.where(df["candidate_side"].eq(df.get("away_team")), "away", "unknown"),
        )
    for col in ["sp_throws", "opp_sp_throws"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown")

    reports = {
        "summary": pd.DataFrame([_summarize(df)]),
        "by_edge": _write_report(df, "edge_bucket", "edge_diagnostics_by_edge.csv"),
        "by_odds": _write_report(df, "odds_bucket", "edge_diagnostics_by_odds.csv"),
        "by_month": _write_report(df, "month", "edge_diagnostics_by_month.csv"),
        "by_season_phase": _write_report(df, "season_phase", "edge_diagnostics_by_season_phase.csv"),
        "by_side": _write_report(df, "home_or_away", "edge_diagnostics_by_side.csv"),
        "by_favdog": _write_report(df, "favdog", "edge_diagnostics_by_favdog.csv"),
        "by_sp_throws": _write_report(df, "sp_throws", "edge_diagnostics_by_sp_throws.csv"),
        "by_opp_sp_throws": _write_report(df, "opp_sp_throws", "edge_diagnostics_by_opp_sp_throws.csv"),
    }
    if "clv_prob" in df.columns and df["clv_prob"].notna().any():
        df["clv_bucket"] = pd.cut(df["clv_prob"], CLV_BINS, labels=CLV_LABELS, right=False)
        reports["by_clv"] = _write_report(df, "clv_bucket", "edge_diagnostics_by_clv.csv")
    reports["summary"].to_csv(
        os.path.join(DATA_DIR, "edge_diagnostics_summary.csv"),
        index=False,
    )
    return reports


def print_report(name: str, report: pd.DataFrame) -> None:
    cols = [c for c in report.columns if c not in {"avg_model_prob", "avg_vegas_prob"}]
    print(f"\n{name}")
    print(report[cols].to_string(index=False))


if __name__ == "__main__":
    reports = build_reports()
    print_report("Summary", reports["summary"])
    print_report("By edge bucket", reports["by_edge"])
    print_report("By odds bucket", reports["by_odds"])
    print_report("By season phase", reports["by_season_phase"])
    print_report("By side", reports["by_side"])
    print_report("By favorite/underdog", reports["by_favdog"])
    print_report("By SP handedness", reports["by_sp_throws"])
    print_report("By opposing SP handedness", reports["by_opp_sp_throws"])
    print(f"\nReports written under {DATA_DIR}")
