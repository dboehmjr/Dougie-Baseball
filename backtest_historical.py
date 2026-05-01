"""
Historical backtest: 2015–2021 using SBRO real closing moneylines.

Uses WALK-FORWARD validation to avoid data leakage:
  - To predict year Y, only train on years < Y

Bet sizing: fractional Kelly, compounding off current bankroll.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Kelly betting: stake = KELLY_FRAC * (edge/net_odds) * current_bankroll
KELLY_FRAC   = 0.25     # quarter-Kelly to reduce variance
MAX_BET_PCT  = 0.10     # cap any single bet at 10% of current bankroll
MIN_EDGE     = 0.03     # only bet if model edge > 3%
STARTING_BK  = 100.0

XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "use_label_encoder": False,
    "random_state":     42,
    "n_jobs":           2,
}


def american_to_implied(ml: float) -> float:
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)


def devig_prob(home_ml: float, away_ml: float) -> tuple[float, float]:
    h, a = american_to_implied(home_ml), american_to_implied(away_ml)
    total = h + a
    return h / total, a / total


def kelly_stake(p: float, ml: float, bankroll: float,
                ref_bankroll: float = STARTING_BK) -> float:
    """
    Non-compounding fractional Kelly: stake scales with edge/odds but is
    always a fraction of the STARTING bankroll, not the running one.
    This prevents bankroll runaway from compounding small edges over 2000+ bets.
    P&L still compounds — the bankroll column shows realistic growth/decline.
    """
    if ml == 0:
        return 0.0
    net  = (ml / 100) if ml > 0 else (100 / abs(ml))
    edge = p * net - (1 - p)
    if edge <= 0 or net <= 0:
        return 0.0
    k = KELLY_FRAC * (edge / net)
    return round(min(k, MAX_BET_PCT) * ref_bankroll, 2)


def pnl_from_ml(stake: float, ml: float, won: bool) -> float:
    if won:
        return stake * (ml / 100) if ml > 0 else stake * (100 / abs(ml))
    return -stake


def train_for_year(feat_df: pd.DataFrame, target_year: int,
                   feature_cols: list[str]) -> object | None:
    train = feat_df[feat_df["year"] < target_year]
    if len(train) < 500:
        return None
    X, y = train[feature_cols], train["home_win"]
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(**XGB_PARAMS)),
    ])
    calibrated = CalibratedClassifierCV(base_estimator=pipeline, method="isotonic", cv=3)
    calibrated.fit(X, y)
    return calibrated


def run_backtest() -> pd.DataFrame:
    print("Loading features.csv…")
    feat_df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"), parse_dates=["Date"])

    print("Loading odds data…")
    odds_frames = [pd.read_csv(os.path.join(DATA_DIR, "historical_odds.csv"), parse_dates=["date"])]
    odds_2026_path = os.path.join(DATA_DIR, "action_network_odds_2026.csv")
    if os.path.exists(odds_2026_path):
        o26 = pd.read_csv(odds_2026_path, parse_dates=["game_date"])
        o26 = o26.rename(columns={"game_date": "date"})
        odds_frames.append(o26[["date", "home_team", "away_team", "home_ml", "away_ml"]])
    odds_df = pd.concat(odds_frames, ignore_index=True)
    odds_df["year"] = odds_df["date"].dt.year

    saved = joblib.load(os.path.join(MODEL_DIR, "win_prob_model.pkl"))
    # Exclude prediction-time-only features not in features.csv
    SKIP = {"vegas_home_prob", "home_lineup_ops", "away_lineup_ops", "lineup_ops_diff",
            "home_batting_ops_vs_sp", "away_batting_ops_vs_sp", "batting_ops_vs_sp_diff"}
    static_features = [c for c in saved["features"] if c in feat_df.columns and c not in SKIP]

    odds_years = sorted(odds_df["year"].unique())
    backtest_years = [y for y in odds_years if y >= 2016]
    print(f"Walk-forward years: {backtest_years}")
    print(f"Training features : {len(static_features)}")

    feat_df["date_key"] = feat_df["Date"].dt.date
    odds_df["date_key"] = odds_df["date"].dt.date

    bankroll  = STARTING_BK
    ledger    = []

    for year in backtest_years:
        print(f"\n  Training on years < {year}…", end=" ", flush=True)
        model = train_for_year(feat_df, year, static_features)
        if model is None:
            print("skipped")
            continue
        print("done")

        year_games = feat_df[feat_df["year"] == year].copy()
        year_odds  = odds_df[odds_df["year"] == year].copy()

        year_games["model_home_prob"] = model.predict_proba(year_games[static_features])[:, 1]

        merged = year_games.merge(
            year_odds[["date_key", "home_team", "home_ml", "away_ml"]],
            on=["date_key", "home_team"],
        ).dropna(subset=["home_ml", "away_ml"])
        # Valid American moneylines are always >= +100 or <= -100
        merged = merged[
            (merged["home_ml"].abs() >= 100) & (merged["away_ml"].abs() >= 100)
        ]

        print(f"  {year}: {len(merged)} games matched", end="")

        yr_bets, yr_wins, yr_pnl = 0, 0, 0.0

        for _, row in merged.sort_values("Date").iterrows():
            mp_h = row["model_home_prob"]
            mp_a = 1 - mp_h
            vp_h, vp_a = devig_prob(row["home_ml"], row["away_ml"])

            edge_h = mp_h - vp_h
            edge_a = mp_a - vp_a

            if edge_h >= edge_a and edge_h > MIN_EDGE:
                bet_side, bet_ml, mp, vp, edge = "home", row["home_ml"], mp_h, vp_h, edge_h
                won = bool(row["home_win"] == 1)
            elif edge_a > edge_h and edge_a > MIN_EDGE:
                bet_side, bet_ml, mp, vp, edge = "away", row["away_ml"], mp_a, vp_a, edge_a
                won = bool(row["home_win"] == 0)
            else:
                continue

            stake     = kelly_stake(mp, bet_ml, bankroll)
            if stake <= 0:
                continue
            pnl       = pnl_from_ml(stake, bet_ml, won)
            bankroll += pnl
            yr_bets  += 1
            yr_wins  += int(won)
            yr_pnl   += pnl

            ledger.append({
                "date":       row["Date"].date(),
                "year":       int(year),
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "bet_side":   bet_side,
                "model_prob": round(mp, 4),
                "vegas_prob": round(vp, 4),
                "edge":       round(edge, 4),
                "moneyline":  int(bet_ml),
                "stake":      round(stake, 2),
                "won":        int(won),
                "pnl":        round(pnl, 2),
                "bankroll":   round(bankroll, 2),
            })

        yr_wagered = sum(r["stake"] for r in ledger[-yr_bets:]) if yr_bets > 0 else 1
        roi = 100 * yr_pnl / yr_wagered if yr_bets > 0 else 0
        print(f"  →  {yr_bets} bets  {yr_wins}/{yr_bets} ({100*yr_wins/yr_bets:.1f}%)  "
              f"P&L: ${yr_pnl:+.2f}  ROI: {roi:+.1f}%  BK: ${bankroll:.2f}")

    return pd.DataFrame(ledger)


def print_summary(ledger: pd.DataFrame) -> None:
    if ledger.empty:
        print("No bets placed.")
        return

    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD BACKTEST — {ledger['year'].min()}–{ledger['year'].max()}")
    print(f"  Starting bankroll : ${STARTING_BK:.2f}")
    print(f"  Bet sizing        : {KELLY_FRAC:.0%} Kelly, max {MAX_BET_PCT:.0%}/bet (compounding)")
    print(f"  Min edge required : {MIN_EDGE*100:.0f}%  (model prob − Vegas implied)")
    print(f"{'='*62}")

    for year, grp in ledger.groupby("year"):
        bets  = len(grp)
        wins  = grp["won"].sum()
        pnl   = grp["pnl"].sum()
        stake = grp["stake"].sum()
        roi   = 100 * pnl / stake if stake > 0 else 0
        end_bk = grp["bankroll"].iloc[-1]
        print(f"  {year}  {bets:>4} bets  {wins}/{bets} ({100*wins/bets:.1f}%)  "
              f"P&L: ${pnl:+7.2f}  ROI: {roi:+5.1f}%  BK: ${end_bk:.2f}")

    tot_bets  = len(ledger)
    tot_wins  = ledger["won"].sum()
    tot_pnl   = ledger["pnl"].sum()
    tot_stake = ledger["stake"].sum()
    final_bk  = ledger["bankroll"].iloc[-1]
    overall   = 100 * tot_pnl / tot_stake if tot_stake > 0 else 0
    n_years   = ledger["year"].nunique()

    print(f"{'─'*62}")
    print(f"  TOTAL  {tot_bets:>5} bets  {tot_wins}/{tot_bets} ({100*tot_wins/tot_bets:.1f}%)  "
          f"P&L: ${tot_pnl:+.2f}  ROI: {overall:+.1f}%")
    print(f"  ${STARTING_BK:.2f} → ${final_bk:.2f}  "
          f"({100*(final_bk/STARTING_BK-1):+.1f}% total, ~{100*(final_bk/STARTING_BK-1)/n_years:+.1f}%/yr)")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    ledger = run_backtest()
    print_summary(ledger)

    out = os.path.join(DATA_DIR, "backtest_historical_results.csv")
    ledger.to_csv(out, index=False)
    print(f"Full ledger → {out}")
