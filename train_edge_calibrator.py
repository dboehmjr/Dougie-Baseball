"""
Train a market-edge calibration layer from walk-forward candidate predictions.

The base baseball model estimates win probability. This layer learns how much
of the raw model-vs-market edge survives out of sample:

    calibrated_prob = vegas_prob + f(raw_model_prob - vegas_prob)

where f is a monotonic isotonic regression fit on walk-forward seasons only.
Save artifact:
  models/edge_calibrator.pkl
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_PATH = os.path.join(DATA_DIR, "walkforward_results.csv")
OUT_PATH = os.path.join(MODEL_DIR, "edge_calibrator.pkl")


def american_profit(stake: float, ml: float, won: bool) -> float:
    if won:
        return stake * (ml / 100) if ml > 0 else stake * (100 / abs(ml))
    return -stake


def fit_calibrator(path: str = RESULTS_PATH) -> dict:
    df = pd.read_csv(path)
    required = {"edge", "model_prob", "vegas_prob", "won", "odds"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns needed for calibration: {missing}")

    train = df.dropna(subset=["edge", "model_prob", "vegas_prob", "won", "odds"]).copy()
    train["won"] = train["won"].astype(bool).astype(float)
    train["residual"] = train["won"] - train["vegas_prob"]

    iso = IsotonicRegression(y_min=-0.20, y_max=0.20, increasing=True, out_of_bounds="clip")
    iso.fit(train["edge"], train["residual"])

    train["calibrated_prob"] = np.clip(train["vegas_prob"] + iso.predict(train["edge"]), 0.01, 0.99)
    train["calibrated_edge"] = train["calibrated_prob"] - train["vegas_prob"]

    return {
        "model": iso,
        "input": "raw_edge",
        "target": "realized_result_minus_vegas_prob",
        "n_rows": int(len(train)),
        "edge_min": float(train["edge"].min()),
        "edge_max": float(train["edge"].max()),
    }


def simulate_thresholds(artifact: dict, path: str = RESULTS_PATH) -> pd.DataFrame:
    iso = artifact["model"]
    df = pd.read_csv(path)
    df = df.dropna(subset=["edge", "vegas_prob", "won", "odds"]).copy()
    df["won"] = df["won"].astype(bool)
    df["calibrated_prob"] = np.clip(df["vegas_prob"] + iso.predict(df["edge"]), 0.01, 0.99)
    df["calibrated_edge"] = df["calibrated_prob"] - df["vegas_prob"]

    rows = []
    for min_edge in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        bets = df[df["calibrated_edge"] >= min_edge].copy()
        if bets.empty:
            rows.append({"min_edge": min_edge, "bets": 0})
            continue
        stake = 1.0
        pnl = sum(american_profit(stake, r["odds"], bool(r["won"])) for _, r in bets.iterrows())
        rows.append({
            "min_edge": min_edge,
            "bets": len(bets),
            "win_rate": bets["won"].mean(),
            "avg_raw_edge": bets["edge"].mean(),
            "avg_calibrated_edge": bets["calibrated_edge"].mean(),
            "flat_stake_roi": pnl / len(bets),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = fit_calibrator()
    joblib.dump(artifact, OUT_PATH)
    print(f"Saved edge calibrator to {OUT_PATH}")
    print(f"Training rows: {artifact['n_rows']:,}")
    print("\nThreshold simulation (flat $1 stakes):")
    print(simulate_thresholds(artifact).to_string(index=False))
