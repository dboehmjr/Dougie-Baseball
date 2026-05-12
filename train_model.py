"""
Train XGBoost win-probability models with cross-validated hyperparameter search.

XGBoost captures non-linear interactions (e.g. park_factor × SP ERA) that logistic
regression cannot.  We use TimeSeriesSplit so validation folds always use past data
to predict future games — no leakage.

Steps:
  1. Load features.csv
  2. Time-based train/test split (2015-2024 train, 2025 test)
  3. Impute missing values
  4. RandomizedSearchCV over XGBoost hyperparameters using TimeSeriesSplit
  5. Calibrate probabilities with chronological isotonic regression
  6. Evaluate: accuracy, log loss, Brier score, ROC AUC, calibration curve
  7. Compare against logistic regression baseline
  8. Save market-independent and market-aware model artifacts
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import database as db
from feature_defaults import apply_feature_defaults

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
    # In-season rolling SP ERA (last 3 starts) — recent form signal
    "home_sp_last3_era",
    "away_sp_last3_era",
    "sp_last3_era_diff",
    "home_bullpen_inseason_era",
    "away_bullpen_inseason_era",
    "bullpen_inseason_era_diff",
    # Contextual game-state impact form (Retrosheet/MLB RE24-style deltas).
    # Ablation testing favored pitcher-side state impact; offense RE24 remains
    # in the feature table for diagnostics but is excluded from training.
    "home_sp_re24_last3",
    "away_sp_re24_last3",
    "sp_re24_diff",
    "home_bullpen_re24_15d",
    "away_bullpen_re24_15d",
    "bullpen_re24_diff",
    # Rest & travel
    "home_days_rest",
    "away_days_rest",
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
    # Prior-season platoon quality against opposing SP hand
    "home_batting_ops_vs_sp",
    "away_batting_ops_vs_sp",
    "batting_ops_vs_sp_diff",
    "home_platoon_advantage",
    "away_platoon_advantage",
    "platoon_advantage_diff",
    # Historical confirmed starting-lineup quality vs opposing SP hand
    "home_lineup_ops_vs_sp",
    "away_lineup_ops_vs_sp",
    "lineup_ops_vs_sp_diff",
    # Lineup OPS vs LHP/RHP separately (handedness split signal)
    "home_lineup_ops_vs_lhp",
    "home_lineup_ops_vs_rhp",
    "away_lineup_ops_vs_lhp",
    "away_lineup_ops_vs_rhp",
    "lineup_ops_vs_lhp_diff",
    "lineup_ops_vs_rhp_diff",
    # Prior-season Pythagorean win% (stable team quality anchor)
    "home_prior_win_pct",
    "away_prior_win_pct",
    "prior_win_pct_diff",
    # Current-season running win% + delta vs prior-year baseline
    "home_season_win_pct",
    "away_season_win_pct",
    "season_win_pct_diff",
    "home_season_win_pct_delta",
    "away_season_win_pct_delta",
    "season_win_pct_delta_diff",
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
    # SP pitch stuff (prior-season FanGraphs: velocity, xFIP)
    "home_sp_fbv",
    "away_sp_fbv",
    "home_sp_xfip",
    "away_sp_xfip",
    "sp_xfip_diff",
    # SP sample size — PA batted in prior season (low = stats less reliable)
    "home_sp_pa",
    "away_sp_pa",
    "sp_pa_diff",
    # SP workload: outs thrown in last start
    "home_sp_outs_last",
    "away_sp_outs_last",
    "sp_outs_last_diff",
    # Prior-season Statcast power metrics (barrel rate, hard hit%)
    "barrel_pct_diff",
    "home_hard_hit_pct",
    "away_hard_hit_pct",
    "hard_hit_pct_diff",
    # Home/away venue split run differential
    "home_home_rd",
    "away_away_rd",
    "rd_venue_diff",
    # Calendar timing: known pre-game; helps early-season calibration
    "game_month",
    "game_day_of_year",
    "is_early_season",
    # Market-aware feature. Excluded from the primary edge model.
    "vegas_home_prob",
]

MARKET_FEATURE_COLS = ["vegas_home_prob"]
MARKET_INDEPENDENT_FEATURE_COLS = [
    c for c in FEATURE_COLS if c not in MARKET_FEATURE_COLS
]

TARGET_COL = "home_win"
TEST_YEAR  = 2025   # train on 2015–2024, test on 2025


# ---------------------------------------------------------------------------
# Load + split
# ---------------------------------------------------------------------------

def load_and_split(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = db.read_table_or_csv("features", path, parse_dates=["Date"])
    df = apply_feature_defaults(df, FEATURE_COLS)
    # Only use completed games for training/testing (exclude ongoing 2026 season)
    train = df[df["year"] < TEST_YEAR].copy()
    test  = df[df["year"] == TEST_YEAR].copy()
    print(f"Years in dataset: {sorted(df['year'].unique())}")
    return train, test


# ---------------------------------------------------------------------------
# XGBoost pipeline with hyperparameter search
# ---------------------------------------------------------------------------

XGB_PARAM_GRID = {
    "n_estimators":      [200, 400, 600],
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     [0.01, 0.05, 0.1],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0],
    "min_child_weight":  [1, 3, 5],
    "gamma":             [0, 0.1, 0.3],
}


def make_sample_weights(years: pd.Series) -> np.ndarray:
    """
    Upweight recent seasons to reduce distribution shift from rule changes.
    2023+ (shift ban + pitch clock): 3x
    2022: 2x
    2020-2021: 1.5x
    pre-2020: 1x
    """
    w = np.ones(len(years), dtype=float)
    w[years.values >= 2023] = 3.0
    w[years.values == 2022] = 2.0
    w[(years.values >= 2020) & (years.values <= 2021)] = 1.5
    return w


def build_xgb_model() -> XGBClassifier:
    # XGBoost handles NaN natively (no imputer needed) — cleaner and lets
    # sample_weight flow without Pipeline routing complications.
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )


def _calibrated_classifier(estimator, n_splits: int = 5) -> CalibratedClassifierCV:
    """Build a chronological calibrator; sklearn renamed this arg in newer releases."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    try:
        return CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv=cv)


def tune_and_fit(X_train: pd.DataFrame, y_train: pd.Series,
                  train_years: pd.Series | None = None):
    """
    RandomizedSearchCV with TimeSeriesSplit + sample weights for recent seasons.
    Tune on log loss because this model is used as a probability forecaster;
    AUC only measures ranking and can preserve overconfident probabilities.
    XGBoost handles NaN natively — no imputer pipeline needed.
    """
    weights = make_sample_weights(train_years) if train_years is not None else None

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        build_xgb_model(),
        param_distributions=XGB_PARAM_GRID,
        n_iter=15,
        scoring="neg_log_loss",
        cv=tscv,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )

    print("Running hyperparameter search (15 iterations × 3 folds)...")
    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    search.fit(X_train, y_train, **fit_kwargs)

    best_idx = search.best_index_
    cv_stds  = search.cv_results_["std_test_score"]
    print(f"\nBest params:   {search.best_params_}")
    print(f"Best CV log loss: {-search.best_score_:.4f}  ± {cv_stds[best_idx]:.4f}")

    # Calibrate with isotonic regression, passing same weights
    best      = search.best_estimator_
    calibrated = _calibrated_classifier(best)
    print("\nCalibrating probabilities (with sample weights)...")
    calibrated.fit(X_train, y_train, **fit_kwargs)

    return calibrated


# ---------------------------------------------------------------------------
# Logistic regression baseline (for comparison)
# ---------------------------------------------------------------------------

def build_lr_baseline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   _calibrated_classifier(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
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
                    y_test: pd.Series, suffix: str = "") -> None:
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
    out_name = f"evaluation{suffix}.png"
    fig.savefig(os.path.join(PLOTS_DIR, out_name), dpi=150)
    plt.close()
    print(f"\nPlots saved to plots/{out_name}")


def _get_base_estimator(calibrated_model):
    clf = calibrated_model.calibrated_classifiers_[0]
    return getattr(clf, "estimator", getattr(clf, "base_estimator", None))


def plot_feature_importance(model, feature_cols: list[str], suffix: str = "") -> None:
    try:
        # Pull importance from one of the calibrated XGB estimators
        base = _get_base_estimator(model)
        xgb_clf = base.named_steps["model"] if hasattr(base, "named_steps") else base
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
        out_name = f"feature_importance{suffix}.png"
        fig.savefig(os.path.join(PLOTS_DIR, out_name), dpi=150)
        plt.close()
        print(f"Feature importance plot saved to plots/{out_name}")

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

    available_all = [c for c in FEATURE_COLS if c in train.columns]
    missing = [c for c in FEATURE_COLS if c not in train.columns]
    coverage_pct = len(available_all) / len(FEATURE_COLS) * 100
    print(f"Features: {len(available_all)}/{len(FEATURE_COLS)} ({coverage_pct:.0f}% coverage)")
    if missing:
        print(f"  Missing features: {missing}")
    print(f"Train: {len(train):,} games | Test: {len(test):,} games")

    y_train = train[TARGET_COL]
    y_test  = test[TARGET_COL]

    specs = [
        {
            "name": "market_independent",
            "label": "Market-independent",
            "features": [c for c in MARKET_INDEPENDENT_FEATURE_COLS if c in train.columns],
            "path": os.path.join(MODEL_DIR, "win_prob_model.pkl"),
            "suffix": "",
        },
        {
            "name": "market_aware",
            "label": "Market-aware",
            "features": [c for c in FEATURE_COLS if c in train.columns],
            "path": os.path.join(MODEL_DIR, "win_prob_market_model.pkl"),
            "suffix": "_market",
        },
    ]

    for spec in specs:
        feat_cols = spec["features"]
        print(f"\n--- Training {spec['label']} model ({len(feat_cols)} features) ---")
        X_train = train[feat_cols]
        X_test  = test[feat_cols]

        xgb_model = tune_and_fit(X_train, y_train, train_years=train["year"])
        metrics_xgb = compute_metrics(
            f"XGBoost ({spec['label']}, tuned + calibrated)",
            xgb_model, X_test, y_test,
        )

        print("\nFitting logistic regression baseline...")
        lr_model = build_lr_baseline()
        lr_model.fit(X_train, y_train)
        metrics_lr = compute_metrics(
            f"Logistic Regression ({spec['label']} baseline)",
            lr_model, X_test, y_test,
        )

        plot_evaluation(metrics_xgb, metrics_lr, y_test, suffix=spec["suffix"])
        plot_feature_importance(xgb_model, feat_cols, suffix=spec["suffix"])

        joblib.dump({
            "pipeline": xgb_model,
            "features": feat_cols,
            "model_type": spec["name"],
            "uses_market": "vegas_home_prob" in feat_cols,
            "target": TARGET_COL,
        }, spec["path"])
        print(f"\n{spec['label']} XGBoost model saved to {spec['path']}")
