"""
Diagnose which feature ranges coincide with profitable 2025 bets.

Outputs:
  data/feature_roi_diagnostics_by_bucket.csv
  data/feature_roi_diagnostics_summary.csv

For home-away diff features, values are flipped for away bets so positive
means "favors the side being bet" whenever possible.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_2025_kelly.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "win_prob_model.pkl")

BUCKET_LABELS = ["low", "mid", "high"]


def _safe_roi(pnl: float, stake: float) -> float:
    return float(pnl / stake) if stake and stake > 0 else np.nan


def _bucket_summary(group: pd.DataFrame) -> pd.Series:
    bets = group[group["stake"] > 0]
    stake = float(bets["stake"].sum())
    pnl = float(bets["game_pnl"].sum())
    return pd.Series({
        "games": len(group),
        "bets": len(bets),
        "wins": int(bets["won"].sum()) if len(bets) else 0,
        "win_rate": float(bets["won"].mean()) if len(bets) else np.nan,
        "avg_value": float(group["feature_value"].mean()),
        "avg_edge": float(bets["edge"].mean()) if len(bets) else np.nan,
        "staked": stake,
        "pnl": pnl,
        "roi": _safe_roi(pnl, stake),
    })


def _feature_value(df: pd.DataFrame, feature: str) -> pd.Series:
    raw = pd.to_numeric(df[feature], errors="coerce")
    is_away_bet = df["bet_side"].eq(df["away_team"])

    if feature.endswith("_diff"):
        return raw.where(~is_away_bet, -raw)
    if feature.startswith("home_"):
        away_feature = "away_" + feature.removeprefix("home_")
        if away_feature in df.columns:
            return pd.to_numeric(df[feature], errors="coerce").where(
                ~is_away_bet,
                pd.to_numeric(df[away_feature], errors="coerce"),
            )
    if feature.startswith("away_"):
        home_feature = "home_" + feature.removeprefix("away_")
        if home_feature in df.columns:
            return pd.to_numeric(df[home_feature], errors="coerce").where(
                ~is_away_bet,
                pd.to_numeric(df[feature], errors="coerce"),
            )
    return raw


def _bucket_feature(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() < 30 or values.nunique(dropna=True) < 3:
        return pd.Series(pd.NA, index=values.index, dtype="object")
    try:
        return pd.qcut(values, q=3, labels=BUCKET_LABELS, duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="object")


def build_feature_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = joblib.load(MODEL_PATH)
    feature_cols = [c for c in bundle["features"] if c != "vegas_home_prob"]

    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    features["date"] = features["Date"].dt.strftime("%Y-%m-%d")
    bt = pd.read_csv(BACKTEST_PATH)
    bt = bt[bt["stake"] > 0].copy()

    merged = bt.merge(
        features.drop(columns=["home_runs", "away_runs"], errors="ignore"),
        on=["date", "home_team", "away_team"],
        how="left",
    )

    bucket_rows = []
    for feature in feature_cols:
        if feature not in merged.columns:
            continue
        values = _feature_value(merged, feature)
        buckets = _bucket_feature(values)
        if buckets.notna().sum() < 30:
            continue

        work = merged.copy()
        work["feature"] = feature
        work["feature_value"] = values
        work["bucket"] = buckets
        work = work.dropna(subset=["feature_value", "bucket"])
        if work.empty:
            continue

        report = (
            work.groupby(["feature", "bucket"], observed=False)
            .apply(_bucket_summary)
            .reset_index()
        )
        bucket_rows.append(report)

    by_bucket = (
        pd.concat(bucket_rows, ignore_index=True)
        if bucket_rows else pd.DataFrame()
    )
    if by_bucket.empty:
        return by_bucket, pd.DataFrame()

    rows = []
    for feature, group in by_bucket.groupby("feature"):
        group = group[group["bets"] >= 8].copy()
        if group.empty:
            continue
        best = group.loc[group["roi"].idxmax()]
        worst = group.loc[group["roi"].idxmin()]
        high = group[group["bucket"].astype(str).eq("high")]
        low = group[group["bucket"].astype(str).eq("low")]
        rows.append({
            "feature": feature,
            "best_bucket": best["bucket"],
            "best_roi": best["roi"],
            "best_bets": int(best["bets"]),
            "worst_bucket": worst["bucket"],
            "worst_roi": worst["roi"],
            "worst_bets": int(worst["bets"]),
            "roi_spread": float(best["roi"] - worst["roi"]),
            "high_minus_low_roi": (
                float(high.iloc[0]["roi"] - low.iloc[0]["roi"])
                if len(high) and len(low) else np.nan
            ),
        })

    summary = pd.DataFrame(rows).sort_values("roi_spread", ascending=False)
    return by_bucket, summary


def main() -> None:
    by_bucket, summary = build_feature_diagnostics()
    by_bucket_path = os.path.join(DATA_DIR, "feature_roi_diagnostics_by_bucket.csv")
    summary_path = os.path.join(DATA_DIR, "feature_roi_diagnostics_summary.csv")
    by_bucket.to_csv(by_bucket_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {by_bucket_path}")
    print(f"Saved {summary_path}")
    if not summary.empty:
        print("\nLargest ROI spreads:")
        cols = ["feature", "best_bucket", "best_roi", "worst_bucket", "worst_roi", "roi_spread"]
        print(summary[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
