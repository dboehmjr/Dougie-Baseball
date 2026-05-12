"""Tune model families and blend weights for the pure-baseball model.

Uses the current MARKET_INDEPENDENT_FEATURE_COLS, which intentionally includes
only the selected SP/bullpen RE24 features.  Produces walk-forward predictions
for a compact set of tree variants and deterministic blend weights.

Outputs:
  data/blend_tuning_walkforward_summary.csv
  data/blend_tuning_walkforward_predictions.csv
"""

from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import database as db
from feature_defaults import apply_feature_defaults
from train_model import MARKET_INDEPENDENT_FEATURE_COLS, make_sample_weights

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRED_PATH = os.path.join(DATA_DIR, "blend_tuning_walkforward_predictions.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "blend_tuning_walkforward_summary.csv")

BASE_SCORE_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
TEST_YEARS = [2024, 2025, 2026]
TEST_2026_END = pd.Timestamp("2026-05-09")


def calibrated(estimator) -> CalibratedClassifierCV:
    cv = TimeSeriesSplit(n_splits=3)
    try:
        return CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv=cv)


def model_specs() -> dict:
    return {
        "xgb_current": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.3, objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "xgb_reg": XGBClassifier(
            n_estimators=350, max_depth=2, learning_rate=0.015,
            subsample=0.85, colsample_bytree=0.75, min_child_weight=8,
            gamma=0.2, reg_lambda=3.0, reg_alpha=0.05,
            objective="binary:logistic", eval_metric="logloss",
            random_state=43, n_jobs=-1,
        ),
        "lgb_current": LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            reg_lambda=1.0, objective="binary", random_state=42,
            n_jobs=-1, verbosity=-1,
        ),
        "lgb_reg": LGBMClassifier(
            n_estimators=450, max_depth=2, learning_rate=0.015,
            subsample=0.85, colsample_bytree=0.75, min_child_samples=60,
            reg_lambda=4.0, reg_alpha=0.1, objective="binary",
            random_state=43, n_jobs=-1, verbosity=-1,
        ),
        "cat_current": CatBoostClassifier(
            iterations=250, depth=4, learning_rate=0.03, l2_leaf_reg=8.0,
            loss_function="Logloss", eval_metric="Logloss",
            random_seed=42, verbose=False, allow_writing_files=False,
        ),
        "cat_reg": CatBoostClassifier(
            iterations=450, depth=3, learning_rate=0.02, l2_leaf_reg=12.0,
            random_strength=1.0, bagging_temperature=0.3,
            loss_function="Logloss", eval_metric="Logloss",
            random_seed=43, verbose=False, allow_writing_files=False,
        ),
    }


def load_features() -> pd.DataFrame:
    features = db.load_features()
    features = apply_feature_defaults(features, MARKET_INDEPENDENT_FEATURE_COLS)
    features = features[features["home_win"].notna()].copy()
    features["Date"] = pd.to_datetime(features["Date"])
    return features.sort_values("Date").reset_index(drop=True)


def test_slice(features: pd.DataFrame, year: int) -> pd.DataFrame:
    test = features[features["year"] == year].copy()
    if year == 2026:
        test = test[test["Date"] <= TEST_2026_END].copy()
    return test


def score_base_year(features: pd.DataFrame, year: int, model_name: str, estimator) -> pd.DataFrame:
    train = features[features["year"] < year].copy()
    test = test_slice(features, year)
    feature_cols = [c for c in MARKET_INDEPENDENT_FEATURE_COLS if c in features.columns]
    X_train = train[feature_cols]
    y_train = train["home_win"].astype(int)
    X_test = test[feature_cols]
    weights = make_sample_weights(train["year"])

    model = calibrated(estimator)
    model.fit(X_train, y_train, sample_weight=weights)
    probs = model.predict_proba(X_test)[:, 1]

    out = test[["Date", "year", "home_team", "away_team", "home_win"]].copy()
    out["base_model"] = model_name
    out["home_prob"] = probs
    return out


def build_base_predictions(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = model_specs()
    for model_name, estimator in specs.items():
        print(f"\nScoring base model: {model_name}")
        for year in BASE_SCORE_YEARS:
            pred = score_base_year(features, year, model_name, estimator)
            rows.append(pred)
            y = pred["home_win"].astype(int)
            p = pred["home_prob"]
            print(
                f"  {year}: acc {accuracy_score(y, p >= 0.5):.3f}, "
                f"logloss {log_loss(y, p, labels=[0, 1]):.4f}, "
                f"AUC {roc_auc_score(y, p):.3f}"
            )
    return pd.concat(rows, ignore_index=True)


def make_game_matrix(base_predictions: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Date", "year", "home_team", "away_team", "home_win"]
    first_model = sorted(base_predictions["base_model"].unique())[0]
    base = base_predictions[base_predictions["base_model"].eq(first_model)][key_cols].copy()
    probs = (
        base_predictions.pivot_table(
            index=["Date", "home_team", "away_team"],
            columns="base_model",
            values="home_prob",
            aggfunc="first",
        )
        .reset_index()
    )
    out = base.merge(probs, on=["Date", "home_team", "away_team"], how="inner")
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)


def candidate_weight_sets(model_names: list[str]) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    for name in model_names:
        weights[f"only_{name}"] = {name: 1.0}
    current = ["xgb_current", "lgb_current", "cat_current"]
    weights["equal_current_trees"] = {m: 1 / 3 for m in current}
    all_models = model_names
    weights["equal_all"] = {m: 1 / len(all_models) for m in all_models}
    weights["cat_current_heavy"] = {"xgb_current": 0.2, "lgb_current": 0.2, "cat_current": 0.6}
    weights["cat_reg_heavy"] = {"xgb_reg": 0.2, "lgb_reg": 0.2, "cat_reg": 0.6}
    weights["cat_pair"] = {"cat_current": 0.5, "cat_reg": 0.5}
    weights["reg_trees_equal"] = {"xgb_reg": 1 / 3, "lgb_reg": 1 / 3, "cat_reg": 1 / 3}

    trio = ["xgb_current", "lgb_current", "cat_current"]
    for wx, wl in itertools.product([0.1, 0.2, 0.3, 0.4], repeat=2):
        wc = round(1.0 - wx - wl, 2)
        if wc < 0.2:
            continue
        weights[f"grid_cur_x{wx:.1f}_l{wl:.1f}_c{wc:.1f}"] = {
            trio[0]: wx, trio[1]: wl, trio[2]: wc,
        }
    return weights


def add_weighted_blends(games: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    model_names = [c for c in model_specs().keys() if c in out.columns]
    for blend_name, weights in candidate_weight_sets(model_names).items():
        prob = np.zeros(len(out))
        total = 0.0
        for model_name, weight in weights.items():
            if model_name not in out.columns:
                continue
            prob += out[model_name].to_numpy() * weight
            total += weight
        if total > 0:
            out[blend_name] = prob / total

    stack_cols = model_names + ["equal_current_trees", "equal_all"]
    out["stacked_logreg"] = np.nan
    for year in sorted(out["year"].unique()):
        train = out[out["year"] < year].copy()
        test = out[out["year"] == year].copy()
        if train.empty:
            out.loc[test.index, "stacked_logreg"] = test["equal_current_trees"]
            continue
        stacker = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
        ])
        weights = np.where(train["year"] >= train["year"].max() - 1, 1.35, 1.0)
        stacker.fit(train[stack_cols], train["home_win"].astype(int), model__sample_weight=weights)
        out.loc[test.index, "stacked_logreg"] = stacker.predict_proba(test[stack_cols])[:, 1]
    return out


def summarize_predictions(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclude = {"Date", "year", "home_team", "away_team", "home_win"}
    prob_cols = [c for c in games.columns if c not in exclude]
    rows = []
    preds = []
    for col in prob_cols:
        for year, group in games[games["year"].isin(TEST_YEARS)].groupby("year"):
            y = group["home_win"].astype(int)
            p = group[col]
            row = {
                "model_name": col,
                "year": int(year),
                "games": len(group),
                "accuracy": accuracy_score(y, p >= 0.5),
                "log_loss": log_loss(y, p, labels=[0, 1]),
                "brier": brier_score_loss(y, p),
                "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
                "avg_confidence": float(np.maximum(p, 1 - p).mean()),
            }
            rows.append(row)
            tmp = group[["Date", "year", "home_team", "away_team", "home_win"]].copy()
            tmp["model_name"] = col
            tmp["model_home_prob"] = p
            preds.append(tmp)
    by_year = pd.DataFrame(rows)
    summary = (
        by_year.groupby("model_name")
        .agg(
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
    return pd.concat(preds, ignore_index=True), summary


def main() -> None:
    features = load_features()
    base_predictions = build_base_predictions(features)
    games = make_game_matrix(base_predictions)
    games = add_weighted_blends(games)
    predictions, summary = summarize_predictions(games)
    predictions.to_csv(PRED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nTop blend/model configs by log loss")
    print(summary.head(20).to_string(index=False, formatters={
        "avg_accuracy": "{:.4f}".format,
        "avg_log_loss": "{:.5f}".format,
        "avg_brier": "{:.5f}".format,
        "avg_auc": "{:.4f}".format,
        "avg_confidence": "{:.4f}".format,
    }))
    print(f"\nSaved -> {PRED_PATH}")
    print(f"Saved -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
