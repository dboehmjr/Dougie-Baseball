"""
Backtest the saved win-probability model against all 2025 MLB games.

Uses the pre-computed features.csv rows (year == 2025) — features were built
using only data available before each game, so there is no leakage.

Output:
  - Overall accuracy, log loss, Brier score, ROC AUC
  - Accuracy by confidence band (how often do we win when we're most sure?)
  - Calibration table (are 60% predictions right ~60% of the time?)
  - Accuracy by month
  - Worst-miss games (biggest upsets we got wrong)
  - Saves backtest_2025_results.csv for further analysis
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score
)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "win_prob_model.pkl")
artifact   = joblib.load(model_path)
pipeline   = artifact["pipeline"]
feat_cols  = artifact["features"]
print(f"Model loaded  ({len(feat_cols)} features)")

# ── Load 2025 games ───────────────────────────────────────────────────────────
features_path = os.path.join(DATA_DIR, "features.csv")
df = pd.read_csv(features_path, parse_dates=["Date"])
games_2025 = df[df["year"] == 2025].copy().sort_values("Date").reset_index(drop=True)
print(f"2025 games    : {len(games_2025):,}")

available = [c for c in feat_cols if c in games_2025.columns]
missing   = [c for c in feat_cols if c not in games_2025.columns]
if missing:
    print(f"Missing cols  : {missing}  (will be imputed)")

X = games_2025[available]
y = games_2025["home_win"]

# ── Predict ───────────────────────────────────────────────────────────────────
probs = pipeline.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

games_2025["prob_home_win"] = probs
games_2025["predicted_winner"] = np.where(preds == 1,
                                           games_2025["home_team"],
                                           games_2025["away_team"])
games_2025["correct"] = (preds == y.values).astype(int)
games_2025["confidence"] = np.maximum(probs, 1 - probs)   # distance from 50/50

# ── Overall metrics ───────────────────────────────────────────────────────────
acc   = accuracy_score(y, preds)
ll    = log_loss(y, probs)
brier = brier_score_loss(y, probs)
auc   = roc_auc_score(y, probs)

print(f"\n{'='*50}")
print(f"  2025 BACKTEST RESULTS  ({len(games_2025):,} games)")
print(f"{'='*50}")
print(f"  Accuracy        : {acc:.1%}  ({int(acc*len(games_2025))}/{len(games_2025)} correct)")
print(f"  ROC AUC         : {auc:.4f}")
print(f"  Log loss        : {ll:.4f}")
print(f"  Brier score     : {brier:.4f}")

# ── Accuracy by confidence band ───────────────────────────────────────────────
bands = [
    (0.50, 0.55, "50–55%  (near coin-flip)"),
    (0.55, 0.60, "55–60%  (slight lean)"),
    (0.60, 0.65, "60–65%  (moderate)"),
    (0.65, 0.70, "65–70%  (strong lean)"),
    (0.70, 1.01, "70%+    (high confidence)"),
]

print(f"\n{'─'*50}")
print(f"  Accuracy by confidence band:")
print(f"  {'Band':<25} {'Games':>6}  {'Correct':>7}  {'Accuracy':>9}")
print(f"  {'─'*25} {'─'*6}  {'─'*7}  {'─'*9}")
for lo, hi, label in bands:
    mask = (games_2025["confidence"] >= lo) & (games_2025["confidence"] < hi)
    subset = games_2025[mask]
    if len(subset) == 0:
        continue
    n = len(subset)
    c = subset["correct"].sum()
    print(f"  {label:<25} {n:>6}  {c:>7}  {c/n:>8.1%}")

# ── Calibration table ─────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  Calibration (predicted vs actual home-win rate):")
print(f"  {'Pred prob bucket':<20} {'Games':>6}  {'Actual win%':>12}")
print(f"  {'─'*20} {'─'*6}  {'─'*12}")
cal_bins = np.arange(0.35, 0.75, 0.05)
for lo in cal_bins:
    hi = lo + 0.05
    mask = (probs >= lo) & (probs < hi)
    n = mask.sum()
    if n == 0:
        continue
    actual = y.values[mask].mean()
    print(f"  {lo:.0%}–{hi:.0%}{'':12} {n:>6}  {actual:>11.1%}")

# ── Accuracy by month ─────────────────────────────────────────────────────────
games_2025["month"] = games_2025["Date"].dt.month
month_names = {4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct"}

print(f"\n{'─'*50}")
print(f"  Accuracy by month:")
print(f"  {'Month':<6} {'Games':>6}  {'Correct':>7}  {'Accuracy':>9}")
print(f"  {'─'*6} {'─'*6}  {'─'*7}  {'─'*9}")
for m in sorted(games_2025["month"].unique()):
    subset = games_2025[games_2025["month"] == m]
    n = len(subset)
    c = subset["correct"].sum()
    print(f"  {month_names.get(m, m):<6} {n:>6}  {c:>7}  {c/n:>8.1%}")

# ── Biggest upsets we got wrong ───────────────────────────────────────────────
wrong = games_2025[games_2025["correct"] == 0].copy()
wrong["surprise"] = wrong["confidence"]   # how wrong we were
worst = wrong.nlargest(10, "surprise")

print(f"\n{'─'*50}")
print(f"  10 biggest upsets (high confidence, wrong pick):")
print(f"  {'Date':<12} {'Matchup':<20} {'Pred%':>6}  {'Predicted':>10}  {'Actual':>10}")
print(f"  {'─'*12} {'─'*20} {'─'*6}  {'─'*10}  {'─'*10}")
for _, row in worst.iterrows():
    matchup = f"{row['away_team']} @ {row['home_team']}"
    pred_team = row["predicted_winner"]
    actual_team = row["home_team"] if row["home_win"] == 1 else row["away_team"]
    conf_pct = row["confidence"]
    date_str = row["Date"].strftime("%Y-%m-%d")
    print(f"  {date_str:<12} {matchup:<20} {conf_pct:>5.1%}  {pred_team:>10}  {actual_team:>10}")

# ── Simulated betting edge ────────────────────────────────────────────────────
# If you bet $1 on every game the model is ≥55% confident on (vs fair odds)
confident = games_2025[games_2025["confidence"] >= 0.55]
if len(confident) > 0:
    bet_acc = confident["correct"].mean()
    print(f"\n{'─'*50}")
    print(f"  Betting edge simulation (≥55% confidence games):")
    print(f"  Games bet      : {len(confident):,}  of  {len(games_2025):,}")
    print(f"  Accuracy       : {bet_acc:.1%}")
    print(f"  Break-even vs -110 juice: 52.4%")
    print(f"  Edge vs break-even: {bet_acc - 0.524:+.1%}")

# ── Save results ──────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "backtest_2025_results.csv")
games_2025[["Date","home_team","away_team","home_win",
            "prob_home_win","predicted_winner","correct","confidence"]
           ].to_csv(out_path, index=False)
print(f"\n{'='*50}")
print(f"  Full results saved to {out_path}")

# ── Calibration plot ──────────────────────────────────────────────────────────
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(7, 5))
frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
ax.plot(mean_pred, frac_pos, "s-", color="#1f77b4", label=f"Model (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",label="Perfect calibration")
ax.set_xlabel("Mean predicted probability (home win)")
ax.set_ylabel("Actual home win rate")
ax.set_title("2025 Backtest — Calibration Curve")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "backtest_2025_calibration.png"), dpi=150)
plt.close()
print(f"  Calibration plot saved to plots/backtest_2025_calibration.png")
