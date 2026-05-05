"""
Backtest the win probability model on the full 2025 season.
2025 is the model's true hold-out year (trained on 2015-2024).
Uses Action Network consensus moneylines from action_network_odds_2022_2025.csv.

Strategy:
  - Bet the side where model probability exceeds devigged Vegas implied prob by > 6%
  - Size with 5% fractional Kelly, capped at 2% of current bankroll
  - Starting bankroll: $100
  - No look-ahead: predictions use only features available before each game

Output: prints summary + saves data/backtest_2025_kelly.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib
from betting_context import attach_starting_pitcher_context, enrich_bet_record, odds_merge_columns
from betting_strategy import choose_bet

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "win_prob_model.pkl")  # market-independent edge model
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
ODDS_PATH     = os.path.join(DATA_DIR, "action_network_odds_2022_2025.csv")

STARTING_BANKROLL = 100.0
KELLY_FRAC        = 0.25
MAX_BET_PCT       = 0.10


def american_to_implied(ml: float) -> float:
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)


def devig_prob(home_ml: float, away_ml: float) -> float:
    ph = american_to_implied(home_ml)
    pa = american_to_implied(away_ml)
    t  = ph + pa
    return ph / t if t > 0 else np.nan


def kelly_stake(p: float, bankroll: float, ml: float) -> float:
    if ml == 0:
        return 0.0
    net  = (ml / 100) if ml > 0 else (100 / abs(ml))
    if net == 0:
        return 0.0
    edge = p * net - (1 - p)
    if edge <= 0:
        return 0.0
    k = KELLY_FRAC * (edge / net)
    return round(min(k, MAX_BET_PCT) * bankroll, 2)


def calc_pnl(stake: float, ml: float, won: bool) -> float:
    if stake == 0:
        return 0.0
    net = (ml / 100) if ml > 0 else (100 / abs(ml))
    return round(stake * net, 2) if won else round(-stake, 2)


def run_backtest() -> pd.DataFrame:
    bundle    = joblib.load(MODEL_PATH)
    pipeline  = bundle["pipeline"]
    feat_cols = bundle["features"]

    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    features = features[features["year"] == 2025].copy()
    print(f"2025 feature rows  : {len(features)}")

    odds = pd.read_csv(ODDS_PATH, dtype={"game_date": str})
    # Filter to 2025 rows only
    odds = odds[odds["game_date"].str.startswith("2025")].copy()
    print(f"2025 odds rows     : {len(odds)}")

    # Filter corrupt odds rows (run-line juice mixed into ML averages)
    def _imp(ml):
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)
    odds["_imp_sum"] = odds["home_ml"].apply(_imp) + odds["away_ml"].apply(_imp)
    odds_clean = odds[(odds["_imp_sum"] >= 1.03) & (odds["_imp_sum"] <= 1.13)].copy()
    print(f"Odds after validity filter: {len(odds_clean)} / {len(odds)}")

    features["game_date"] = features["Date"].dt.strftime("%Y-%m-%d")

    merged = features.merge(
        odds_clean[odds_merge_columns(odds_clean)],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    # Keep feature-engineered scores when present; older features.csv needs odds scores.
    if ("home_runs" not in merged.columns or "away_runs" not in merged.columns) and "home_runs" in odds_clean.columns:
        merged = merged.merge(
            odds_clean[["game_date", "home_team", "away_team", "home_runs", "away_runs"]],
            on=["game_date", "home_team", "away_team"],
            how="left",
        )
    elif "home_runs" not in merged.columns or "away_runs" not in merged.columns:
        merged["home_runs"] = np.nan
        merged["away_runs"] = np.nan

    merged = merged[merged["home_win"].notna()].copy()
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = attach_starting_pitcher_context(merged)

    print(f"Games with outcome : {len(merged)}")
    print(f"Games with odds    : {merged['consensus_prob'].notna().sum()}")

    X = pd.DataFrame(index=merged.index)
    for c in feat_cols:
        X[c] = merged[c] if c in merged.columns else np.nan

    probs = pipeline.predict_proba(X[feat_cols])[:, 1]
    merged["model_home_prob"] = probs

    records = []
    bankroll = STARTING_BANKROLL

    for _, row in merged.iterrows():
        p_home = row["model_home_prob"]
        p_away = 1 - p_home
        home   = row["home_team"]
        away   = row["away_team"]
        date   = row["game_date"]

        home_ml = row.get("home_ml", np.nan)
        away_ml = row.get("away_ml", np.nan)

        base = {
            "date": date, "matchup": f"{away} @ {home}",
            "home_score": row.get("home_runs"), "away_score": row.get("away_runs"),
        }

        if pd.isna(home_ml) or pd.isna(away_ml):
            records.append(enrich_bet_record(
                {**base, "bet_side": None, "stake": 0, "odds": None,
                 "model_prob": None, "vegas_prob": None, "edge": None,
                 "odds_bucket": None, "edge_threshold": None,
                 "policy_reason": "missing odds",
                 "won": None, "game_pnl": 0, "bankroll": bankroll},
                row=row, side=None, model_prob=None, vegas_prob=None, odds=None,
            ))
            continue

        vegas_home = devig_prob(home_ml, away_ml)
        pick = choose_bet(
            home=home, away=away, p_home=p_home,
            home_ml=home_ml, away_ml=away_ml, vegas_home=vegas_home,
            game_date=date,
        )

        if not pick["should_bet"]:
            records.append(enrich_bet_record(
                {**base, "bet_side": None, "stake": 0, "odds": pick["odds"],
                 "model_prob": round(pick["model_prob"], 4),
                 "vegas_prob": round(pick["vegas_prob"], 4),
                 "edge": round(pick["edge"], 4),
                 "odds_bucket": pick["odds_bucket"],
                 "season_phase": pick["season_phase"],
                 "edge_threshold": pick["edge_threshold"],
                 "policy_reason": pick["policy_reason"],
                 "won": None, "game_pnl": 0, "bankroll": bankroll},
                row=row, side=pick["side"], model_prob=pick["model_prob"],
                vegas_prob=pick["vegas_prob"], odds=pick["odds"],
            ))
            continue

        bet_side = pick["side"]
        bet_p = pick["model_prob"]
        pick_ml = pick["odds"]
        bet_edge = pick["edge"]
        vegas_p = pick["vegas_prob"]
        stake    = kelly_stake(bet_p, bankroll, pick_ml)
        won      = (row["home_win"] == 1) if bet_side == home else (row["home_win"] == 0)
        game_pnl = calc_pnl(stake, pick_ml, won)
        bankroll = round(bankroll + game_pnl, 2)

        records.append(enrich_bet_record({**base,
            "bet_side":   bet_side,
            "stake":      stake,
            "odds":       pick_ml,
            "model_prob": round(bet_p, 4),
            "vegas_prob": round(vegas_p, 4),
            "edge":       round(bet_edge, 4),
            "odds_bucket": pick["odds_bucket"],
            "season_phase": pick["season_phase"],
            "edge_threshold": pick["edge_threshold"],
            "policy_reason": pick["policy_reason"],
            "won":        won,
            "game_pnl":   game_pnl,
            "bankroll":   bankroll,
        }, row=row, side=bet_side, model_prob=bet_p, vegas_prob=vegas_p, odds=pick_ml))

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    bets = df[df["stake"] > 0].copy()

    total_games  = len(df)
    total_bets   = len(bets)
    wins         = int(bets["won"].sum())
    losses       = total_bets - wins
    win_rate     = wins / total_bets * 100 if total_bets else 0
    total_staked = bets["stake"].sum()
    total_pnl    = bets["game_pnl"].sum()
    roi          = total_pnl / total_staked * 100 if total_staked else 0
    final_bk     = df["bankroll"].iloc[-1] if not df.empty else STARTING_BANKROLL
    avg_edge     = bets["edge"].mean() * 100 if total_bets else 0
    avg_stake    = bets["stake"].mean() if total_bets else 0

    print(f"\n{'='*50}")
    print(f"  2025 MLB BACKTEST RESULTS (hold-out year)")
    print(f"{'='*50}")
    print(f"  Season games processed : {total_games}")
    print(f"  Games with odds        : {df['vegas_prob'].notna().sum()}")
    print(f"  Bets placed            : {total_bets}")
    print(f"  No-bet games           : {total_games - total_bets}")
    print(f"{'─'*50}")
    print(f"  Win / Loss             : {wins}W – {losses}L")
    print(f"  Win rate               : {win_rate:.1f}%")
    print(f"  Avg edge (bet games)   : {avg_edge:.1f}%")
    print(f"  Avg stake              : ${avg_stake:.2f}")
    print(f"{'─'*50}")
    print(f"  Total staked           : ${total_staked:.2f}")
    print(f"  Total P&L              : ${total_pnl:+.2f}")
    print(f"  ROI                    : {roi:+.1f}%")
    print(f"{'─'*50}")
    print(f"  Starting bankroll      : ${STARTING_BANKROLL:.2f}")
    print(f"  Ending bankroll        : ${final_bk:.2f}")
    print(f"  Bankroll return        : {(final_bk - STARTING_BANKROLL) / STARTING_BANKROLL * 100:+.1f}%")
    print(f"{'='*50}")

    if not bets.empty:
        bets["month"] = pd.to_datetime(bets["date"]).dt.strftime("%Y-%m")
        monthly = (bets.groupby("month")
                       .agg(n=("stake","count"), wins=("won","sum"),
                            staked=("stake","sum"), pnl=("game_pnl","sum"))
                       .reset_index())
        monthly["roi"] = monthly["pnl"] / monthly["staked"] * 100
        print(f"\n  Monthly breakdown:")
        for _, r in monthly.iterrows():
            l = int(r["n"]) - int(r["wins"])
            print(f"    {r['month']}  {int(r['n']):3} bets  "
                  f"{int(r['wins'])}W-{l}L  "
                  f"staked ${r['staked']:6.2f}  "
                  f"P&L ${r['pnl']:+7.2f}  ROI {r['roi']:+5.1f}%")


if __name__ == "__main__":
    print("Loading model and data…")
    df = run_backtest()
    print_summary(df)
    out = os.path.join(DATA_DIR, "backtest_2025_kelly.csv")
    df.to_csv(out, index=False)
    print(f"\n  Full results saved → {out}")
