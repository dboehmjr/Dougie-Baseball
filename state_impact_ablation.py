"""Walk-forward ablations for RE24/state-impact features.

This script answers: which version of the paper-inspired state-impact signal
actually helps the win-probability model?

It evaluates:
  - no RE24 features
  - offense / SP / bullpen groups separately
  - group combinations
  - a small grid of rolling windows

Outputs:
  data/state_impact_ablation_summary.csv
  data/state_impact_ablation_by_year.csv
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

import database as db
from feature_defaults import apply_feature_defaults
from train_model import MARKET_INDEPENDENT_FEATURE_COLS, make_sample_weights

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SUMMARY_PATH = os.path.join(DATA_DIR, "state_impact_ablation_summary.csv")
BY_YEAR_PATH = os.path.join(DATA_DIR, "state_impact_ablation_by_year.csv")

TEST_YEARS = [2024, 2025, 2026]
TEST_2026_END = pd.Timestamp("2026-05-09")

RE24_COLS = [
    "home_offense_re24_15g", "away_offense_re24_15g", "offense_re24_diff",
    "home_sp_re24_last3", "away_sp_re24_last3", "sp_re24_diff",
    "home_bullpen_re24_15d", "away_bullpen_re24_15d", "bullpen_re24_diff",
]
OFFENSE_COLS = ["home_offense_re24_15g", "away_offense_re24_15g", "offense_re24_diff"]
SP_COLS = ["home_sp_re24_last3", "away_sp_re24_last3", "sp_re24_diff"]
BULLPEN_COLS = ["home_bullpen_re24_15d", "away_bullpen_re24_15d", "bullpen_re24_diff"]


def calibrated(estimator) -> CalibratedClassifierCV:
    cv = TimeSeriesSplit(n_splits=3)
    try:
        return CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv=cv)


def model_specs() -> dict:
    return {
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.3, objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            reg_lambda=1.0, objective="binary", random_state=42,
            n_jobs=-1, verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=250, depth=4, learning_rate=0.03, l2_leaf_reg=8.0,
            loss_function="Logloss", eval_metric="Logloss",
            random_seed=42, verbose=False, allow_writing_files=False,
        ),
    }


def load_features() -> pd.DataFrame:
    features = db.load_features()
    features["Date"] = pd.to_datetime(features["Date"])
    features = features[features["home_win"].notna()].copy()
    return features.sort_values("Date").reset_index(drop=True)


def load_plate_appearances() -> pd.DataFrame:
    pa = db.read_table_or_csv(
        "state_impact_plate_appearances",
        os.path.join(DATA_DIR, "state_impact_plate_appearances.csv"),
        parse_dates=["game_date"],
    )
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["pitcher_is_starter"] = pa["pitcher_is_starter"].fillna(False).astype(bool)
    for col in ["offense_re24", "pitching_re24"]:
        pa[col] = pd.to_numeric(pa[col], errors="coerce").fillna(0.0)
    if "game_number" not in pa.columns:
        pa["game_number"] = 1
    pa["game_number"] = pd.to_numeric(pa["game_number"], errors="coerce").fillna(1).astype(int)
    return pa


def _merge_variant(base: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    out = base.drop(columns=[c for c in RE24_COLS if c in base.columns], errors="ignore").copy()
    variant = variant.copy()
    variant["Date"] = pd.to_datetime(variant["game_date"])
    keep = ["Date", "home_team", "away_team", "game_number"] + RE24_COLS
    out = out.merge(variant[keep], on=["Date", "home_team", "away_team", "game_number"], how="left")
    return out


def build_variant_features(pa: pd.DataFrame,
                           offense_games: int = 15,
                           sp_starts: int = 3,
                           bullpen_days: int = 15) -> pd.DataFrame:
    games = (
        pa[["game_date", "home_team", "away_team", "game_number", "game_id"]]
        .drop_duplicates()
        .sort_values(["game_date", "home_team", "away_team", "game_number"])
    )

    offense_game = (
        pa.groupby(["game_date", "game_id", "batting_team"], dropna=False)
        .agg(offense_re24=("offense_re24", "sum"))
        .reset_index()
        .sort_values(["batting_team", "game_date", "game_id"])
    )
    offense_game["team_offense_re24"] = offense_game.groupby("batting_team")["offense_re24"].transform(
        lambda s: s.shift(1).rolling(offense_games, min_periods=max(3, offense_games // 3)).mean()
    )

    starter_game = (
        pa[pa["pitcher_is_starter"]]
        .groupby(["game_date", "game_id", "pitching_team", "pitcher_id"], dropna=False)
        .agg(sp_re24=("pitching_re24", "sum"))
        .reset_index()
        .sort_values(["pitcher_id", "game_date", "game_id"])
    )
    starter_game["sp_re24"] = starter_game.groupby("pitcher_id")["sp_re24"].transform(
        lambda s: s.shift(1).rolling(sp_starts, min_periods=max(2, sp_starts // 2)).mean()
    )

    bullpen_game = (
        pa[~pa["pitcher_is_starter"]]
        .groupby(["pitching_team", "game_date"], dropna=False)
        .agg(bullpen_re24=("pitching_re24", "sum"))
        .reset_index()
        .sort_values(["pitching_team", "game_date"])
    )
    bullpen_parts = []
    for _, grp in bullpen_game.groupby("pitching_team"):
        work = grp.set_index("game_date").sort_index()
        work["bullpen_re24"] = work["bullpen_re24"].rolling(f"{bullpen_days}D", closed="left").mean()
        bullpen_parts.append(work.reset_index())
    bullpen_game = pd.concat(bullpen_parts, ignore_index=True) if bullpen_parts else bullpen_game

    out = games.copy()
    out = out.merge(
        offense_game.rename(columns={"batting_team": "home_team", "team_offense_re24": "home_offense_re24_15g"})
        [["game_date", "game_id", "home_team", "home_offense_re24_15g"]],
        on=["game_date", "game_id", "home_team"], how="left",
    )
    out = out.merge(
        offense_game.rename(columns={"batting_team": "away_team", "team_offense_re24": "away_offense_re24_15g"})
        [["game_date", "game_id", "away_team", "away_offense_re24_15g"]],
        on=["game_date", "game_id", "away_team"], how="left",
    )
    out = out.merge(
        starter_game.rename(columns={"pitching_team": "home_team", "sp_re24": "home_sp_re24_last3"})
        [["game_date", "game_id", "home_team", "home_sp_re24_last3"]],
        on=["game_date", "game_id", "home_team"], how="left",
    )
    out = out.merge(
        starter_game.rename(columns={"pitching_team": "away_team", "sp_re24": "away_sp_re24_last3"})
        [["game_date", "game_id", "away_team", "away_sp_re24_last3"]],
        on=["game_date", "game_id", "away_team"], how="left",
    )
    out = out.merge(
        bullpen_game.rename(columns={"pitching_team": "home_team", "bullpen_re24": "home_bullpen_re24_15d"})
        [["game_date", "home_team", "home_bullpen_re24_15d"]],
        on=["game_date", "home_team"], how="left",
    )
    out = out.merge(
        bullpen_game.rename(columns={"pitching_team": "away_team", "bullpen_re24": "away_bullpen_re24_15d"})
        [["game_date", "away_team", "away_bullpen_re24_15d"]],
        on=["game_date", "away_team"], how="left",
    )
    out["offense_re24_diff"] = out["home_offense_re24_15g"] - out["away_offense_re24_15g"]
    out["sp_re24_diff"] = out["home_sp_re24_last3"] - out["away_sp_re24_last3"]
    out["bullpen_re24_diff"] = out["home_bullpen_re24_15d"] - out["away_bullpen_re24_15d"]
    return out


def apply_ablation(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in RE24_COLS:
        if col not in keep_cols:
            out[col] = 0.0
    return apply_feature_defaults(out, MARKET_INDEPENDENT_FEATURE_COLS)


def test_slice(features: pd.DataFrame, year: int) -> pd.DataFrame:
    test = features[features["year"] == year].copy()
    if year == 2026:
        test = test[test["Date"] <= TEST_2026_END].copy()
    return test


def evaluate_config(features: pd.DataFrame,
                    config_name: str,
                    keep_cols: list[str],
                    model_names: list[str],
                    stage: str) -> list[dict]:
    rows = []
    work = apply_ablation(features, keep_cols)
    feature_cols = [c for c in MARKET_INDEPENDENT_FEATURE_COLS if c in work.columns]
    specs = {k: v for k, v in model_specs().items() if k in model_names}
    for year in TEST_YEARS:
        train = work[work["year"] < year].copy()
        test = test_slice(work, year)
        if train.empty or test.empty:
            continue
        X_train = train[feature_cols]
        y_train = train["home_win"].astype(int)
        X_test = test[feature_cols]
        y_test = test["home_win"].astype(int)
        weights = make_sample_weights(train["year"])
        preds = []
        for _, estimator in specs.items():
            model = calibrated(estimator)
            model.fit(X_train, y_train, sample_weight=weights)
            preds.append(model.predict_proba(X_test)[:, 1])
        p = np.mean(preds, axis=0)
        rows.append({
            "stage": stage,
            "config": config_name,
            "models": "+".join(model_names),
            "year": year,
            "games": len(test),
            "accuracy": accuracy_score(y_test, p >= 0.5),
            "log_loss": log_loss(y_test, p, labels=[0, 1]),
            "brier": brier_score_loss(y_test, p),
            "auc": roc_auc_score(y_test, p) if y_test.nunique() > 1 else np.nan,
            "avg_confidence": float(np.maximum(p, 1 - p).mean()),
        })
    return rows


def summarize(by_year: pd.DataFrame) -> pd.DataFrame:
    return (
        by_year.groupby(["stage", "config", "models"])
        .agg(
            years=("year", "nunique"),
            games=("games", "sum"),
            avg_accuracy=("accuracy", "mean"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier=("brier", "mean"),
            avg_auc=("auc", "mean"),
            avg_confidence=("avg_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["avg_log_loss", "avg_brier", "avg_accuracy"], ascending=[True, True, False])
    )


def main() -> None:
    base = load_features()
    pa = load_plate_appearances()
    variants = [
        ("current_15g_3sp_15bp", 15, 3, 15),
        ("short_7g_3sp_7bp", 7, 3, 7),
        ("medium_15g_5sp_7bp", 15, 5, 7),
        ("long_30g_5sp_15bp", 30, 5, 15),
        ("longsp_15g_8sp_15bp", 15, 8, 15),
    ]
    ablations = [
        ("no_re24", []),
        ("offense_only", OFFENSE_COLS),
        ("sp_only", SP_COLS),
        ("bullpen_only", BULLPEN_COLS),
        ("offense_sp", OFFENSE_COLS + SP_COLS),
        ("offense_bullpen", OFFENSE_COLS + BULLPEN_COLS),
        ("sp_bullpen", SP_COLS + BULLPEN_COLS),
        ("all_re24", RE24_COLS),
    ]

    screen_rows = []
    variant_cache: dict[str, pd.DataFrame] = {}
    for variant_name, off_win, sp_win, bp_win in variants:
        print(f"\nBuilding variant {variant_name}...")
        variant = build_variant_features(pa, off_win, sp_win, bp_win)
        variant_features = _merge_variant(base, variant)
        variant_cache[variant_name] = variant_features
        for ablation_name, keep_cols in ablations:
            config = f"{variant_name}__{ablation_name}"
            print(f"  Screening {config}...")
            screen_rows.extend(evaluate_config(
                variant_features, config, keep_cols,
                model_names=["xgboost"], stage="screen_xgboost",
            ))

    screen_by_year = pd.DataFrame(screen_rows)
    screen_summary = summarize(screen_by_year)
    top_configs = screen_summary.head(8)["config"].tolist()

    confirm_rows = []
    ablation_lookup = {
        f"{variant_name}__{ablation_name}": (variant_name, keep_cols)
        for variant_name, _, _, _ in variants
        for ablation_name, keep_cols in ablations
    }
    print("\nConfirming top configs with full tree blend...")
    for config in top_configs:
        variant_name, keep_cols = ablation_lookup[config]
        print(f"  Confirming {config}...")
        confirm_rows.extend(evaluate_config(
            variant_cache[variant_name], config, keep_cols,
            model_names=["xgboost", "lightgbm", "catboost"], stage="confirm_blend_trees",
        ))

    by_year = pd.concat([screen_by_year, pd.DataFrame(confirm_rows)], ignore_index=True)
    summary = summarize(by_year)
    by_year.to_csv(BY_YEAR_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nTop configs by log loss")
    print(summary.head(15).to_string(index=False, formatters={
        "avg_accuracy": "{:.4f}".format,
        "avg_log_loss": "{:.5f}".format,
        "avg_brier": "{:.5f}".format,
        "avg_auc": "{:.4f}".format,
        "avg_confidence": "{:.4f}".format,
    }))
    print(f"\nSaved -> {SUMMARY_PATH}")
    print(f"Saved -> {BY_YEAR_PATH}")


if __name__ == "__main__":
    main()
