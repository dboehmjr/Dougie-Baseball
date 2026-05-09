"""
Diagnose win-probability calibration without any betting logic.

The reports compare predicted win probability to actual win rate by:
  - probability bucket
  - month
  - team
  - home/away side

Positive calibration_error means the model was underconfident:
actual win rate was higher than predicted probability.

Negative calibration_error means the model was overconfident:
actual win rate was lower than predicted probability.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_PATHS = {
    "independent": os.path.join(MODEL_DIR, "win_prob_model.pkl"),
    "market": os.path.join(MODEL_DIR, "win_prob_market_model.pkl"),
}

PROB_BINS = np.arange(0.0, 1.01, 0.05)


def load_scored_games(model_name: str, years: list[int] | None) -> pd.DataFrame:
    model_path = MODEL_PATHS[model_name]
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feat_cols = artifact["features"]

    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = df[df["home_win"].notna()].copy()
    if years:
        df = df[df["year"].isin(years)].copy()
    if df.empty:
        raise ValueError("No completed games found for the requested years.")

    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        X[col] = df[col] if col in df.columns else np.nan

    df["model_home_prob"] = pipeline.predict_proba(X[feat_cols])[:, 1]
    df["home_win"] = df["home_win"].astype(int)
    df["pred_home_win"] = (df["model_home_prob"] >= 0.5).astype(int)
    df["month"] = df["Date"].dt.strftime("%Y-%m")
    return df


def to_team_perspective(games: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "Date": games["Date"],
        "year": games["year"],
        "month": games["month"],
        "team": games["home_team"],
        "opponent": games["away_team"],
        "side": "home",
        "win_prob": games["model_home_prob"],
        "won": games["home_win"],
    })
    away = pd.DataFrame({
        "Date": games["Date"],
        "year": games["year"],
        "month": games["month"],
        "team": games["away_team"],
        "opponent": games["home_team"],
        "side": "away",
        "win_prob": 1 - games["model_home_prob"],
        "won": 1 - games["home_win"],
    })
    teams = pd.concat([home, away], ignore_index=True)
    teams["prob_bucket"] = pd.cut(
        teams["win_prob"],
        bins=PROB_BINS,
        include_lowest=True,
        right=False,
    )
    return teams


def summarize(group: pd.DataFrame) -> pd.Series:
    n = len(group)
    avg_prob = float(group["win_prob"].mean()) if n else np.nan
    actual = float(group["won"].mean()) if n else np.nan
    error = actual - avg_prob
    return pd.Series({
        "games": n,
        "avg_pred_prob": avg_prob,
        "actual_win_rate": actual,
        "calibration_error": error,
        "abs_calibration_error": abs(error),
        "direction": (
            "underconfident" if error > 0
            else "overconfident" if error < 0
            else "calibrated"
        ),
        "brier": brier_score_loss(group["won"], group["win_prob"]) if n else np.nan,
    })


def grouped_report(df: pd.DataFrame, by: str | list[str], **groupby_kwargs) -> pd.DataFrame:
    return (
        df.groupby(by, **groupby_kwargs)[["win_prob", "won"]]
        .apply(summarize)
        .reset_index()
    )


def build_reports(games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    teams = to_team_perspective(games)
    home_games = pd.DataFrame({
        "month": games["month"],
        "win_prob": games["model_home_prob"],
        "won": games["home_win"],
    })

    metrics = pd.DataFrame([{
        "games": len(games),
        "accuracy": accuracy_score(games["home_win"], games["pred_home_win"]),
        "log_loss": log_loss(games["home_win"], games["model_home_prob"]),
        "brier": brier_score_loss(games["home_win"], games["model_home_prob"]),
        "roc_auc": roc_auc_score(games["home_win"], games["model_home_prob"]),
        "avg_home_prob": games["model_home_prob"].mean(),
        "actual_home_win_rate": games["home_win"].mean(),
    }])

    by_bucket = grouped_report(teams, "prob_bucket", observed=False)
    by_bucket["prob_bucket"] = by_bucket["prob_bucket"].astype(str)

    by_month = (
        grouped_report(home_games, "month")
        .sort_values("month")
    )

    by_team = (
        grouped_report(teams, "team")
        .sort_values(["abs_calibration_error", "games"], ascending=[False, False])
    )

    by_side = grouped_report(teams, "side")

    by_team_side = (
        grouped_report(teams, ["team", "side"])
        .sort_values(["abs_calibration_error", "games"], ascending=[False, False])
    )

    by_month_side = (
        grouped_report(teams, ["month", "side"])
        .sort_values(["month", "side"])
    )

    return {
        "summary": metrics,
        "by_probability_bucket": by_bucket,
        "by_month": by_month,
        "by_team": by_team,
        "by_home_away": by_side,
        "by_team_home_away": by_team_side,
        "by_month_home_away": by_month_side,
    }


def write_reports(reports: dict[str, pd.DataFrame], model_name: str) -> None:
    for name, report in reports.items():
        path = os.path.join(DATA_DIR, f"calibration_{model_name}_{name}.csv")
        report.to_csv(path, index=False)
        print(f"Saved {path}")


def print_report_preview(reports: dict[str, pd.DataFrame]) -> None:
    print("\nSummary")
    print(reports["summary"].round(4).to_string(index=False))

    print("\nProbability buckets")
    cols = ["prob_bucket", "games", "avg_pred_prob", "actual_win_rate", "calibration_error", "brier"]
    print(reports["by_probability_bucket"][cols].round(4).to_string(index=False))

    print("\nMost miscalibrated teams")
    cols = ["team", "games", "avg_pred_prob", "actual_win_rate", "calibration_error", "brier"]
    print(reports["by_team"][cols].head(10).round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Win-probability calibration diagnostics",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PATHS),
        default="independent",
        help="Saved model artifact to diagnose.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[2025, 2026],
        help="Completed seasons/years to score. Default: 2025 2026.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = load_scored_games(args.model, args.years)
    reports = build_reports(games)
    write_reports(reports, args.model)
    print_report_preview(reports)


if __name__ == "__main__":
    main()
