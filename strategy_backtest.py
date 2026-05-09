"""
Strategy backtester — completely separate from model training.

Reads:
  data/model_predictions.csv   — pre-generated model probabilities
  data/action_network_odds_2022_2025.csv
  data/historical_odds.csv     — pre-2022

Tests a Kelly betting strategy and reports per-year and aggregate results.
Tweak the STRATEGY section at the bottom to experiment with different rules.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

PRED_PATH = os.path.join(DATA_DIR, "model_predictions.csv")
AN_PATH   = os.path.join(DATA_DIR, "action_network_odds_2022_2025.csv")
HIST_PATH = os.path.join(DATA_DIR, "historical_odds.csv")

# ── Strategy parameters (edit these) ────────────────────────────────────────
STARTING_BANKROLL = 100.0
MIN_EDGE          = 0.03    # minimum model edge over vegas
KELLY_FRAC        = 0.25    # fractional Kelly
MAX_BET_PCT       = 0.10    # max % of bankroll per bet
MIN_ODDS          = -130    # only bet on teams at these odds or better (underdogs + near even)
MAX_ODDS          = None    # None = no cap (e.g. +300 max)
TEST_YEARS        = [2021, 2022, 2023, 2024, 2025]

# Odds validity bounds per source (keeps vig-inflated edges from exploding Kelly)
ODDS_VIG_BOUNDS   = {
    "pre2022": (1.01, 1.05),   # historical_odds.csv ~1.02 vig
    "2022plus": (1.03, 1.13),  # Action Network ~1.06 vig
}
# ────────────────────────────────────────────────────────────────────────────


def american_to_implied(ml: float) -> float:
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)


def devig(home_ml: float, away_ml: float) -> float:
    ph = american_to_implied(home_ml)
    pa = american_to_implied(away_ml)
    t  = ph + pa
    return ph / t if t > 0 else np.nan


def kelly_stake(p: float, bankroll: float, ml: float) -> float:
    net = (ml / 100) if ml > 0 else (100 / abs(ml))
    if net <= 0:
        return 0.0
    edge = p * net - (1 - p)
    if edge <= 0:
        return 0.0
    return round(min(KELLY_FRAC * (edge / net), MAX_BET_PCT) * bankroll, 2)


def calc_pnl(stake: float, ml: float, won: bool) -> float:
    net = (ml / 100) if ml > 0 else (100 / abs(ml))
    return round(stake * net, 2) if won else round(-stake, 2)


def load_odds() -> pd.DataFrame:
    """Load and combine all odds sources into one DataFrame."""
    # Action Network 2022–2025
    an = pd.read_csv(AN_PATH, dtype={"game_date": str})
    an["game_date"] = pd.to_datetime(an["game_date"])

    # Historical pre-2022
    hist = pd.read_csv(HIST_PATH, dtype={"date": str})
    hist = hist.rename(columns={"date": "game_date"})
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    hist["consensus_prob"] = hist.apply(
        lambda r: devig(r["home_ml"], r["away_ml"])
        if pd.notna(r.get("home_ml")) and pd.notna(r.get("away_ml")) else np.nan,
        axis=1,
    )

    combined = pd.concat([an, hist], ignore_index=True)

    # Validity filter by source
    def _imp(ml):
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)

    combined["_imp_sum"] = combined["home_ml"].apply(_imp) + combined["away_ml"].apply(_imp)
    combined["_year"]    = combined["game_date"].dt.year
    lo = combined["_year"].map(lambda y: 1.01 if y < 2022 else 1.03)
    hi = combined["_year"].map(lambda y: 1.05 if y < 2022 else 1.13)
    combined = combined[(combined["_imp_sum"] >= lo) & (combined["_imp_sum"] <= hi)]

    return combined[["game_date", "home_team", "away_team",
                      "home_ml", "away_ml", "consensus_prob"]].dropna(subset=["consensus_prob"])


def passes_odds_filter(ml: float) -> bool:
    """Return True if the odds pass the MIN_ODDS / MAX_ODDS filter."""
    if MIN_ODDS is not None and ml < MIN_ODDS:
        return False
    if MAX_ODDS is not None and ml > MAX_ODDS:
        return False
    return True


def run_year(year_preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    preds = year_preds.copy()
    preds["game_date"] = pd.to_datetime(preds["Date"])

    merged = preds.merge(
        odds[["game_date", "home_team", "away_team", "home_ml", "away_ml", "consensus_prob"]],
        on=["game_date", "home_team", "away_team"], how="left",
    )
    merged = merged[merged["home_win"].notna()].sort_values("Date").reset_index(drop=True)

    records = []
    bankroll = STARTING_BANKROLL

    for _, row in merged.iterrows():
        p_home   = row["model_home_prob"]
        p_away   = 1 - p_home
        home_ml  = row.get("home_ml", np.nan)
        away_ml  = row.get("away_ml", np.nan)
        date     = row["game_date"].strftime("%Y-%m-%d")
        matchup  = f"{row['away_team']} @ {row['home_team']}"

        base = {"date": date, "matchup": matchup}

        if pd.isna(home_ml) or pd.isna(away_ml):
            records.append({**base, "bet_side": None, "stake": 0, "game_pnl": 0, "bankroll": bankroll})
            continue

        vegas_home = devig(home_ml, away_ml)
        vegas_away = 1 - vegas_home
        edge_home  = p_home - vegas_home
        edge_away  = p_away - vegas_away

        # Pick the side with the larger edge
        if edge_home >= edge_away:
            bet_side, bet_p, pick_ml, bet_edge, vegas_p = row["home_team"], p_home, home_ml, edge_home, vegas_home
        else:
            bet_side, bet_p, pick_ml, bet_edge, vegas_p = row["away_team"], p_away, away_ml, edge_away, vegas_away

        # Apply strategy filters
        if bet_edge <= MIN_EDGE or not passes_odds_filter(pick_ml):
            records.append({**base, "bet_side": None, "stake": 0,
                             "model_prob": round(bet_p, 4), "vegas_prob": round(vegas_p, 4),
                             "edge": round(bet_edge, 4), "game_pnl": 0, "bankroll": bankroll})
            continue

        stake      = kelly_stake(bet_p, bankroll, pick_ml)
        won        = (row["home_win"] == 1) if bet_side == row["home_team"] else (row["home_win"] == 0)
        game_pnl   = calc_pnl(stake, pick_ml, won)
        flat_pnl   = calc_pnl(1.0, pick_ml, won)   # $1 flat bet for ROI comparison
        bankroll   = round(bankroll + game_pnl, 2)

        records.append({**base,
            "bet_side":   bet_side,
            "stake":      stake,
            "odds":       pick_ml,
            "model_prob": round(bet_p, 4),
            "vegas_prob": round(vegas_p, 4),
            "edge":       round(bet_edge, 4),
            "won":        won,
            "game_pnl":   game_pnl,
            "flat_pnl":   flat_pnl,
            "bankroll":   bankroll,
        })

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, year: int) -> dict:
    bets = df[df["stake"] > 0]
    n    = len(bets)
    if n == 0:
        return {"year": year, "bets": 0, "wins": 0, "losses": 0,
                "win_pct": 0, "avg_edge": 0, "staked": 0, "pnl": 0,
                "roi": 0, "flat_roi": 0,
                "final_bk": df["bankroll"].iloc[-1] if not df.empty else STARTING_BANKROLL}
    wins      = int(bets["won"].sum())
    staked    = bets["stake"].sum()
    pnl       = bets["game_pnl"].sum()
    flat_pnl  = bets["flat_pnl"].sum() if "flat_pnl" in bets.columns else np.nan
    roi       = pnl / staked * 100 if staked else 0
    flat_roi  = flat_pnl / n * 100 if n else 0   # flat $1/bet ROI
    final     = df["bankroll"].iloc[-1]
    return {
        "year":     year,
        "games":    len(df),
        "bets":     n,
        "wins":     wins,
        "losses":   n - wins,
        "win_pct":  round(wins / n * 100, 1),
        "avg_edge": round(bets["edge"].mean() * 100, 1),
        "staked":   round(staked, 2),
        "pnl":      round(pnl, 2),
        "roi":      round(roi, 1),
        "flat_roi": round(flat_roi, 1),
        "final_bk": round(final, 2),
    }


if __name__ == "__main__":
    print(f"Strategy parameters:")
    print(f"  MIN_EDGE   : {MIN_EDGE:.0%}")
    print(f"  MIN_ODDS   : {MIN_ODDS}")
    print(f"  MAX_ODDS   : {MAX_ODDS}")
    print(f"  KELLY_FRAC : {KELLY_FRAC}")
    print(f"  MAX_BET_PCT: {MAX_BET_PCT:.0%}")
    print()

    print("Loading predictions and odds...")
    preds = pd.read_csv(PRED_PATH, parse_dates=["Date"])
    odds  = load_odds()
    print(f"  Predictions: {len(preds):,} games ({preds['year'].min()}–{preds['year'].max()})")
    print(f"  Odds rows  : {len(odds):,}")

    all_records = []
    summaries   = []

    for year in TEST_YEARS:
        year_preds = preds[preds["year"] == year].copy()
        year_odds  = odds[odds["game_date"].dt.year == year].copy()
        df = run_year(year_preds, year_odds)   # bankroll resets to $100 each year
        df["year"] = year
        all_records.append(df)
        s = summarize(df, year)
        summaries.append(s)
        print(f"  {year}: {s['bets']} bets  W/L: {s['wins']}-{s['losses']}  "
              f"Win%: {s['win_pct']}%  ROI: {s['roi']:+.1f}%  "
              f"Flat ROI: {s['flat_roi']:+.1f}%  BK: ${s['final_bk']:.2f}")

    print(f"\n{'='*70}")
    print(f"  {'Year':<6} {'Bets':>5} {'W–L':>10} {'Win%':>6} {'Avg Edge':>9} {'Kelly ROI':>10} {'Flat ROI':>9} {'Final BK':>10}")
    print(f"  {'─'*68}")

    total_bets = total_wins = total_staked = total_pnl = total_flat = 0
    for s in summaries:
        wl = f"{s['wins']}–{s['losses']}"
        print(f"  {s['year']:<6} {s['bets']:>5} {wl:>10} {s['win_pct']:>5.1f}% "
              f"{s['avg_edge']:>8.1f}% {s['roi']:>+9.1f}% {s['flat_roi']:>+8.1f}% "
              f"${s['final_bk']:>9.2f}")
        total_bets   += s["bets"]
        total_wins   += s["wins"]
        total_staked += s["staked"]
        total_pnl    += s["pnl"]
        total_flat   += s.get("flat_roi", 0)

    total_losses  = total_bets - total_wins
    total_kelly   = total_pnl / total_staked * 100 if total_staked else 0
    avg_flat_roi  = total_flat / len(summaries)
    print(f"  {'─'*68}")
    print(f"  {'TOTAL':<6} {total_bets:>5} {total_wins}–{total_losses} "
          f"{total_wins/total_bets*100:>5.1f}%"
          f"           {total_kelly:>+9.1f}%  avg:{avg_flat_roi:>+7.1f}%")
    print(f"{'='*70}")

    all_df = pd.concat(all_records, ignore_index=True)
    out = os.path.join(DATA_DIR, "strategy_backtest_results.csv")
    all_df.to_csv(out, index=False)
    print(f"\nFull results saved → {out}")
