"""
Compare ROC AUC: market-independent vs market-informed model.

Market-independent: vegas_home_prob is NaN (imputed to median) — current setup.
Market-informed:    vegas_home_prob populated from Action Network consensus_prob.

Both models train on 2015–2024, test on 2025.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FIXED_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.3,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, n_jobs=-1,
)

TEST_YEAR = 2025

def build_model():
    xgb = XGBClassifier(**FIXED_PARAMS)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", xgb)])
    return CalibratedClassifierCV(pipe, method="isotonic", cv=3)

def metrics(name, model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  ROC AUC    : {roc_auc_score(y, probs):.4f}")
    print(f"  Log Loss   : {log_loss(y, probs):.4f}")
    print(f"  Brier Score: {brier_score_loss(y, probs):.4f}")
    print(f"  Accuracy   : {accuracy_score(y, preds):.4f}")

# Load features + model feature list
print("Loading features...")
features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"), parse_dates=["Date"])
bundle    = joblib.load(os.path.join(MODEL_DIR, "win_prob_model.pkl"))
feat_cols = bundle["features"]

train = features[features["year"] < TEST_YEAR].copy()
test  = features[features["year"] == TEST_YEAR].copy()
print(f"Train: {len(train):,} games | Test: {len(test):,} games")
print(f"Feature cols: {len(feat_cols)}")

# ── Model 1: market-independent (vegas_home_prob = NaN → imputed to median) ──
X_train_blind = train[feat_cols].copy()
X_test_blind  = test[feat_cols].copy()
y_train = train["home_win"].values
y_test  = test["home_win"].values

print("\nTraining market-independent model...")
m_blind = build_model()
m_blind.fit(X_train_blind, y_train)

# ── Merge Vegas odds into features ──────────────────────────────────────────
print("\nMerging Vegas odds...")

def american_to_implied(ml):
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)

def devig(home_ml, away_ml):
    ph = american_to_implied(home_ml)
    pa = american_to_implied(away_ml)
    t  = ph + pa
    return ph / t if t > 0 else np.nan

# Load Action Network (2022–2025)
an = pd.read_csv(os.path.join(DATA_DIR, "action_network_odds_2022_2025.csv"), dtype={"game_date": str})
an["game_date"] = pd.to_datetime(an["game_date"])
an["_vegas_prob"] = an["consensus_prob"]

# Load historical xlsx odds for 2015–2021
hist_dfs = []
for yr in range(2015, 2022):
    xlsx = os.path.join(DATA_DIR, f"mlb-odds-{yr}.xlsx")
    if not os.path.exists(xlsx):
        continue
    df = pd.read_excel(xlsx)
    # Standardize columns — typical format: Date, Rot, VH, Team, Pitcher, ML
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    hist_dfs.append(df)

def build_vegas_lookup(features_df):
    """Merge AN odds; return DataFrame with vegas_home_prob filled."""
    df = features_df.copy()
    df["_date_str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    an_sub = an[["game_date", "home_team", "away_team", "_vegas_prob"]].copy()
    an_sub["_date_str"] = an_sub["game_date"].dt.strftime("%Y-%m-%d")
    
    merged = df.merge(
        an_sub[["_date_str", "home_team", "away_team", "_vegas_prob"]],
        on=["_date_str", "home_team", "away_team"], how="left"
    )
    merged["vegas_home_prob"] = merged["_vegas_prob"].combine_first(merged.get("vegas_home_prob", pd.Series(dtype=float)))
    return merged

train_w = build_vegas_lookup(train)
test_w  = build_vegas_lookup(test)

train_filled = (train_w["vegas_home_prob"].notna().sum())
test_filled  = (test_w["vegas_home_prob"].notna().sum())
print(f"  Train odds filled: {train_filled:,}/{len(train):,} ({train_filled/len(train)*100:.0f}%)")
print(f"  Test  odds filled: {test_filled:,}/{len(test):,}  ({test_filled/len(test)*100:.0f}%)")

X_train_mkt = train_w[feat_cols].copy()
X_test_mkt  = test_w[feat_cols].copy()

# ── Model 2: market-informed ─────────────────────────────────────────────────
print("\nTraining market-informed model...")
m_mkt = build_model()
m_mkt.fit(X_train_mkt, y_train)

# ── Results ──────────────────────────────────────────────────────────────────
metrics("Market-Independent  (vegas = NaN → median)", m_blind, X_test_blind, y_test)
metrics("Market-Informed     (vegas_home_prob filled)", m_mkt, X_test_mkt, y_test)

# How much of the test set actually has odds?
print(f"\n  Test games with odds: {test_filled}/{len(test)}")
print(f"  (Subset AUC below uses only games WITH odds, both models)")

mask = test_w["vegas_home_prob"].notna().values
if mask.sum() > 0:
    probs_blind_sub = m_blind.predict_proba(X_test_blind[mask])[:, 1]
    probs_mkt_sub   = m_mkt.predict_proba(X_test_mkt[mask])[:, 1]
    y_sub           = y_test[mask]
    print(f"\n  Subset ({mask.sum()} games with odds):")
    print(f"    Market-Independent AUC : {roc_auc_score(y_sub, probs_blind_sub):.4f}")
    print(f"    Market-Informed    AUC : {roc_auc_score(y_sub, probs_mkt_sub):.4f}")
    
    # Vegas-only model for reference
    vegas_probs = test_w.loc[mask, "vegas_home_prob"].values
    print(f"    Vegas-only         AUC : {roc_auc_score(y_sub, vegas_probs):.4f}")

