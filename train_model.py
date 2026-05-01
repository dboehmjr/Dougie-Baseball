"""
Train an XGBoost win-probability model with cross-validated hyperparameter search.

XGBoost captures non-linear interactions (e.g. park_factor × SP ERA) that logistic
regression cannot.  We use TimeSeriesSplit so validation folds always use past data
to predict future games — no leakage.

Steps:
  1. Load features.csv
  2. Time-based train/test split (2015-2023 train, 2024 test)
  3. Impute missing values
  4. RandomizedSearchCV over XGBoost hyperparameters using TimeSeriesSplit
  5. Calibrate probabilities with isotonic regression
  6. Evaluate: accuracy, log loss, Brier score, ROC AUC, calibration curve
  7. Compare against logistic regression baseline
  8. Save model to models/win_prob_model.pkl
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    RocCurveDisplay,
)
from xgboost import XGBClassifier

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "home_rolling_rd",
    "away_rolling_rd",
    "rd_diff",
    "home_rolling_rs",
    "away_rolling_rs",
    "rs_diff",
    # Short-term form: 7-game rolling run differential
    "home_last7_rd",
    "away_last7_rd",
    "last7_rd_diff",
    # Momentum (7-game minus 15-game run diff)
    "home_momentum",
    "away_momentum",
    "momentum_diff",
    # Win/loss streak going into the game
    "home_streak",
    "away_streak",
    "streak_diff",
    "home_sp_era_adj",
    "away_sp_era_adj",
    "sp_era_adj_diff",
    "home_sp_inseason_era",
    "away_sp_inseason_era",
    "sp_inseason_era_diff",
    "home_bullpen_inseason_era",
    "away_bullpen_inseason_era",
    "bullpen_inseason_era_diff",
    "park_factor",
    # Rest & travel
    "home_days_rest",
    "away_days_rest",
    "rest_diff",
    "away_travel_miles",
    "travel_diff",
    # Head-to-head history (last 10 meetings)
    "h2h_home_win_rate",
    "h2h_home_run_diff",
    # Bullpen usage / fatigue (outs thrown in last 3 calendar days)
    "home_bullpen_outs_3d",
    "away_bullpen_outs_3d",
    "bullpen_usage_diff",
    # Prior-season team batting quality (PA-weighted OPS)
    "home_team_ops",
    "away_team_ops",
    "ops_diff",
    # Game-time weather at home park
    "temp_f",
    "wind_speed_mph",
    "wind_to_cf",
    "humidity_pct",
    # Umpire tendency (historical run factor vs. league average)
    "ump_run_factor",
    # Injured List counts (monthly snapshots, forward-filled to game date)
    "home_il_count",
    "away_il_count",
    "il_diff",
    # IL quality: prior-year WAR of players on the IL
    "home_il_war",
    "away_il_war",
    "il_war_diff",
    # Park dimensions (static per home team)
    "park_lf_dist",
    "park_cf_dist",
    "park_rf_dist",
    "park_lf_wall_ht",
    "park_altitude_ft",
    # SP pitch stuff (FanGraphs season-level: velocity, whiff rate, K%, xFIP)
    "home_sp_fbv",
    "away_sp_fbv",
    "sp_fbv_diff",
    "home_sp_swstr",
    "away_sp_swstr",
    "sp_swstr_diff",
    "home_sp_k_pct",
    "away_sp_k_pct",
    "sp_k_pct_diff",
    "home_sp_xfip",
    "away_sp_xfip",
    "sp_xfip_diff",
    # Prior-season Statcast power metrics (barrel rate, hard hit%)
    "home_barrel_pct",
    "away_barrel_pct",
    "barrel_pct_diff",
    "home_hard_hit_pct",
    "away_hard_hit_pct",
    "hard_hit_pct_diff",
    # Prediction-time-only features (NaN in training; imputed to median by SimpleImputer)
    # Vegas consensus moneyline probability (devigged)
    "vegas_home_prob",
    # Confirmed lineup average OPS
    "home_lineup_ops",
    "away_lineup_ops",
    "lineup_ops_diff",
    # Batting OPS vs SP handedness (from L/R splits)
    "home_batting_ops_vs_sp",
    "away_batting_ops_vs_sp",
    "batting_ops_vs_sp_diff",
]

TARGET_COL = "home_win"
TEST_YEAR  = 2025   # train on 2015–2024, test on 2025


# ---------------------------------------------------------------------------
# Load + split
# ---------------------------------------------------------------------------

def load_and_split(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["Date"])
    # Only use completed games for training/testing (exclude ongoing 2026 season)
    train = df[df["year"] < TEST_YEAR].copy()
    test  = df[df["year"] == TEST_YEAR].copy()
    print(f"Years in dataset: {sorted(df['year'].unique())}")
    return train, test


# ---------------------------------------------------------------------------
# XGBoost pipeline with hyperparameter search
# ---------------------------------------------------------------------------

XGB_PARAM_GRID = {
    "model__n_estimators":      [200, 400, 600],
    "model__max_depth":         [3, 4, 5, 6],
    "model__learning_rate":     [0.01, 0.05, 0.1],
    "model__subsample":         [0.7, 0.8, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 1.0],
    "model__min_child_weight":  [1, 3, 5],
    "model__gamma":             [0, 0.1, 0.3],
}


def build_xgb_pipeline() -> Pipeline:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=2,          # limit to 2 cores — prevents overheating
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   xgb),
    ])


def tune_and_fit(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    RandomizedSearchCV with TimeSeriesSplit — later folds always predict
    games that come after the training window.
    """
    base_pipeline = build_xgb_pipeline()

    tscv = TimeSeriesSplit(n_splits=3)   # reduced from 5 to save time/memory

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=XGB_PARAM_GRID,
        n_iter=15,             # reduced from 30 — still finds good params
        scoring="roc_auc",
        cv=tscv,
        random_state=42,
        n_jobs=1,              # run folds sequentially — prevents memory spikes
        verbose=1,
    )

    print("Running hyperparameter search (15 iterations × 3 folds)...")
    search.fit(X_train, y_train)

    best_idx = search.best_index_
    cv_scores = search.cv_results_["mean_test_score"]
    cv_stds   = search.cv_results_["std_test_score"]
    print(f"\nBest params:   {search.best_params_}")
    print(f"Best CV AUC:   {search.best_score_:.4f}  ± {cv_stds[best_idx]:.4f}")

    # Calibrate the best estimator with isotonic regression
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(
        base_estimator=best,
        method="isotonic",
        cv=5,
    )
    print("\nCalibrating probabilities...")
    calibrated.fit(X_train, y_train)

    return calibrated


# ---------------------------------------------------------------------------
# Logistic regression baseline (for comparison)
# ---------------------------------------------------------------------------

def build_lr_baseline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   CalibratedClassifierCV(
            base_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            method="sigmoid", cv=5,
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(name: str, model, X_test, y_test) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    m = {
        "accuracy":    accuracy_score(y_test, preds),
        "log_loss":    log_loss(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),
        "roc_auc":     roc_auc_score(y_test, probs),
        "_probs":      probs,
    }
    print(f"\n=== {name} ===")
    for k, v in m.items():
        if not k.startswith("_"):
            print(f"  {k:<15} {v:.4f}")
    return m


def plot_evaluation(metrics_xgb: dict, metrics_lr: dict,
                    y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Calibration curves
    for label, m, marker in [("XGBoost", metrics_xgb, "s"), ("LogReg", metrics_lr, "^")]:
        fp, mp = calibration_curve(y_test, m["_probs"], n_bins=10)
        axes[0].plot(mp, fp, f"{marker}-", label=label)
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ROC curves
    for label, m in [("XGBoost", metrics_xgb), ("LogReg", metrics_lr)]:
        RocCurveDisplay.from_predictions(
            y_test, m["_probs"], name=label, ax=axes[1]
        )
    axes[1].set_title("ROC Curve")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "evaluation.png"), dpi=150)
    plt.close()
    print("\nPlots saved to plots/evaluation.png")


def plot_feature_importance(model, feature_cols: list[str]) -> None:
    try:
        # Pull importance from one of the calibrated XGB estimators
        xgb_clf = model.calibrated_classifiers_[0].base_estimator.named_steps["model"]
        importance = xgb_clf.feature_importances_
        n = min(len(importance), len(feature_cols))
        imp_df = pd.DataFrame({
            "feature":    feature_cols[:n],
            "importance": importance[:n],
        }).sort_values("importance")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#1f77b4")
        ax.set_xlabel("Feature importance (gain)")
        ax.set_title("XGBoost Feature Importance")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        print("Feature importance plot saved to plots/feature_importance.png")

        print("\nFeature importances:")
        for _, row in imp_df.sort_values("importance", ascending=False).iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")
    except Exception as exc:
        print(f"Could not plot feature importance: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features_path = os.path.join(DATA_DIR, "features.csv")
    print(f"Loading features from {features_path}")
    train, test = load_and_split(features_path)

    available = [c for c in FEATURE_COLS if c in train.columns]
    missing = [c for c in FEATURE_COLS if c not in train.columns]
    coverage_pct = len(available) / len(FEATURE_COLS) * 100
    print(f"Features: {len(available)}/{len(FEATURE_COLS)} ({coverage_pct:.0f}% coverage)")
    if missing:
        print(f"  Missing features: {missing}")
    print(f"Train: {len(train):,} games | Test: {len(test):,} games")

    X_train, y_train = train[available], train[TARGET_COL]
    X_test,  y_test  = test[available],  test[TARGET_COL]

    # --- XGBoost ---
    xgb_model = tune_and_fit(X_train, y_train)
    metrics_xgb = compute_metrics("XGBoost (tuned + calibrated)", xgb_model, X_test, y_test)

    # --- Logistic regression baseline ---
    print("\nFitting logistic regression baseline...")
    lr_model = build_lr_baseline()
    lr_model.fit(X_train, y_train)
    metrics_lr = compute_metrics("Logistic Regression (baseline)", lr_model, X_test, y_test)

    # --- Plots ---
    plot_evaluation(metrics_xgb, metrics_lr, y_test)
    plot_feature_importance(xgb_model, available)

    # --- Save XGBoost as primary model ---
    model_path = os.path.join(MODEL_DIR, "win_prob_model.pkl")
    joblib.dump({"pipeline": xgb_model, "features": available}, model_path)
    print(f"\nXGBoost model saved to {model_path}")
