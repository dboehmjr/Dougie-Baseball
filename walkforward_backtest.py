"""
Walk-forward validation: train on 2015–(Y-1), test on Y, for Y in 2021–2025.

Uses fixed hyperparams from the most recent full RandomizedSearchCV run so each
fold trains in seconds rather than minutes.  Odds sources:
  - 2021        : data/historical_odds.csv  (raw home_ml / away_ml)
  - 2022–2025   : data/action_network_odds_2022_2025.csv  (consensus_prob)

Output: prints per-year + aggregate summary, saves data/walkforward_results.csv
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from betting_context import attach_starting_pitcher_context, enrich_bet_record, odds_merge_columns
from betting_strategy import BettingPolicy, PolicyRule, choose_bet, save_policy, season_phase

warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FEAT_PATH  = os.path.join(DATA_DIR, "features.csv")
AN_PATH    = os.path.join(DATA_DIR, "action_network_odds_2022_2025.csv")
HIST_PATH  = os.path.join(DATA_DIR, "historical_odds.csv")

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# Fixed hyperparams from last RandomizedSearchCV run
FIXED_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.3,
    eval_metric="logloss",
    random_state=42, n_jobs=-1,
)

STARTING_BANKROLL = 100.0
KELLY_FRAC        = 0.25
MAX_BET_PCT       = 0.10
EDGE_THRESHOLDS   = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
ODDS_BUCKETS      = ["<-180", "-180:-140", "-140:-115", "-115:+100", "+100:+130", "+130:+170", "+170+"]
SEASON_PHASES     = ["early", "mid", "late"]
MIN_TUNE_BETS     = 40
MIN_RECENT_BETS   = 30
MIN_YEAR_BETS     = 20
MIN_TUNE_ROI      = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def flat_profit(odds: float, won: bool) -> float:
    if won:
        return (odds / 100) if odds > 0 else (100 / abs(odds))
    return -1.0


def tune_policy(prior_records: list[pd.DataFrame], test_year: int) -> tuple[BettingPolicy, pd.DataFrame]:
    """
    Tune one edge threshold per odds bucket using all prior out-of-sample seasons.
    No season-phase splits — pool all phases within each bucket.
    Only buckets that are profitable in the majority of prior years are kept.
    Falls back to +130:+170 > 6% when insufficient prior data.
    """
    FALLBACK_RULE   = PolicyRule("+130:+170", 0.06)
    MIN_BUCKET_BETS = 30    # minimum pooled bets per bucket to tune
    ALL_THRESHOLDS  = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    fallback = BettingPolicy(rules=(FALLBACK_RULE,), name="fallback_underdog_130_170_6pct")

    if not prior_records:
        return fallback, pd.DataFrame([{
            "test_year": test_year, "odds_bucket": FALLBACK_RULE.bucket,
            "edge_threshold": FALLBACK_RULE.min_edge, "prior_bets": 0,
            "prior_flat_roi": np.nan, "policy_name": fallback.name,
            "reason": "bootstrap: no prior seasons",
        }])

    prior = pd.concat(prior_records, ignore_index=True)
    prior = prior.dropna(subset=["edge", "odds", "won", "odds_bucket"]).copy()
    prior["won"] = prior["won"].astype(bool)
    prior_years = sorted(prior["year"].dropna().astype(int).unique())
    n_years = len(prior_years)

    rules: list[PolicyRule] = []
    rows: list[dict] = []

    for bucket in ODDS_BUCKETS:
        bucket_df = prior[prior["odds_bucket"] == bucket]
        best = None

        for threshold in ALL_THRESHOLDS:
            bets = bucket_df[bucket_df["edge"] > threshold].copy()
            n = len(bets)
            if n < MIN_BUCKET_BETS:
                continue

            pnl = sum(flat_profit(float(r["odds"]), bool(r["won"])) for _, r in bets.iterrows())
            roi = pnl / n

            profitable_years = sum(
                1 for _, yb in bets.groupby("year")
                if sum(flat_profit(float(r["odds"]), bool(r["won"])) for _, r in yb.iterrows()) / len(yb) > 0
            )

            score = roi * min(1.0, n / 300)
            candidate = {
                "test_year": test_year,
                "odds_bucket": bucket,
                "edge_threshold": threshold,
                "prior_bets": n,
                "prior_flat_roi": round(roi, 4),
                "profitable_years": profitable_years,
                "total_years": n_years,
                "score": round(score, 4),
                "reason": "candidate",
            }

            # Require positive ROI and profitable in most prior years
            if roi > 0 and profitable_years >= max(1, n_years - 1):
                if best is None or score > best["score"]:
                    best = candidate

        if best is None:
            rows.append({
                "test_year": test_year, "odds_bucket": bucket,
                "edge_threshold": np.nan, "prior_bets": int(len(bucket_df)),
                "prior_flat_roi": np.nan, "reason": "no profitable threshold found",
            })
            continue

        rules.append(PolicyRule(bucket, best["edge_threshold"]))
        rows.append({**best, "reason": "selected"})

    name = f"walkforward_tuned_through_{test_year - 1}"
    rows_df = pd.DataFrame(rows)

    if not rules:
        rows_df["policy_name"] = fallback.name
        return fallback, rows_df

    policy = BettingPolicy(rules=tuple(rules), name=name)
    rows_df["policy_name"] = name
    return policy, rows_df


# ---------------------------------------------------------------------------
# Odds loader
# ---------------------------------------------------------------------------

def load_odds(year: int) -> pd.DataFrame:
    """Return odds DataFrame with columns: game_date, home_team, away_team,
    home_ml, away_ml, consensus_prob — filtered and validity-checked."""

    def _imp(ml):
        return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)

    if year >= 2022:
        df = pd.read_csv(AN_PATH, dtype={"game_date": str})
        df = df[df["game_date"].str.startswith(str(year))].copy()
    else:
        df = pd.read_csv(HIST_PATH, dtype={"date": str})
        df = df[df["date"].str.startswith(str(year))].copy()
        df = df.rename(columns={"date": "game_date"})
        df["consensus_prob"] = df.apply(
            lambda r: devig_prob(r["home_ml"], r["away_ml"])
            if pd.notna(r.get("home_ml")) and pd.notna(r.get("away_ml")) else np.nan,
            axis=1,
        )

    # Validity filter — historical_odds.csv has ~1.02 vig (already partially devigged),
    # while Action Network averages ~1.06. Use source-appropriate bounds.
    df["_imp_sum"] = df["home_ml"].apply(_imp) + df["away_ml"].apply(_imp)
    lo, hi = (1.01, 1.05) if year < 2022 else (1.03, 1.13)
    df = df[(df["_imp_sum"] >= lo) & (df["_imp_sum"] <= hi)].copy()
    return df[odds_merge_columns(df)].dropna(subset=["consensus_prob"])


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def train_fold(train_df: pd.DataFrame, feat_cols: list[str]) -> CalibratedClassifierCV:
    X = train_df[feat_cols].copy()
    y = train_df["home_win"].values

    xgb = XGBClassifier(**FIXED_PARAMS)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   xgb),
    ])
    cv = TimeSeriesSplit(n_splits=3)
    try:
        cal = CalibratedClassifierCV(pipe, method="isotonic", cv=cv)
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=pipe, method="isotonic", cv=cv)
    cal.fit(X, y)
    return cal


# ---------------------------------------------------------------------------
# Kelly simulation for one test year
# ---------------------------------------------------------------------------

def run_year(test_df: pd.DataFrame, odds: pd.DataFrame,
             model: CalibratedClassifierCV, feat_cols: list[str],
             policy: BettingPolicy) -> pd.DataFrame:

    test_df = test_df.copy()
    test_df["game_date"] = pd.to_datetime(test_df["Date"]).dt.strftime("%Y-%m-%d")

    merged = test_df.merge(
        odds[odds_merge_columns(odds)],
        on=["game_date", "home_team", "away_team"], how="left",
    )
    merged = merged[merged["home_win"].notna()].sort_values("Date").reset_index(drop=True)
    merged = attach_starting_pitcher_context(merged)

    X = pd.DataFrame(index=merged.index)
    for c in feat_cols:
        X[c] = merged[c] if c in merged.columns else np.nan
    probs = model.predict_proba(X[feat_cols])[:, 1]
    merged["model_home_prob"] = probs

    records = []
    bankroll = STARTING_BANKROLL

    for _, row in merged.iterrows():
        p_home = row["model_home_prob"]
        p_away = 1 - p_home
        home, away = row["home_team"], row["away_team"]
        date = row["game_date"]
        home_ml = row.get("home_ml", np.nan)
        away_ml = row.get("away_ml", np.nan)

        base = {
            "date": date,
            "matchup": f"{away} @ {home}",
            "home_team": home,
            "away_team": away,
            "model_home_prob": round(p_home, 4),
        }

        if pd.isna(home_ml) or pd.isna(away_ml):
            records.append({**base, "candidate_side": None, "bet_side": None,
                            "stake": 0, "odds": None, "model_prob": None,
                            "vegas_prob": None, "edge": None, "is_favorite": None,
                            "season_phase": season_phase(date),
                            "odds_bucket": None, "edge_threshold": None,
                            "policy_reason": "missing odds",
                            "policy_name": policy.name,
                            "won": None, "game_pnl": 0, "bankroll": bankroll,
                            "side_home_away": None, "sp_name": None,
                            "opp_sp_name": None, "sp_throws": None,
                            "opp_sp_throws": None, "close_prob": np.nan,
                            "clv_prob": np.nan})
            continue

        vegas_home = devig_prob(home_ml, away_ml)
        pick = choose_bet(
            home=home, away=away, p_home=p_home,
            home_ml=home_ml, away_ml=away_ml, vegas_home=vegas_home,
            policy=policy, game_date=date,
        )
        candidate_side = pick["side"]
        candidate_p = pick["model_prob"]
        candidate_ml = pick["odds"]
        candidate_edge = pick["edge"]
        candidate_vegas_p = pick["vegas_prob"]
        candidate_won = row["home_win"] == 1 if candidate_side == home else row["home_win"] == 0
        is_favorite = bool(candidate_ml < 0) if pd.notna(candidate_ml) else None

        if not pick["should_bet"]:
            rec = {**base,
                            "candidate_side": candidate_side,
                            "bet_side": None,
                            "stake": 0,
                            "odds": candidate_ml,
                            "model_prob": round(candidate_p, 4),
                            "vegas_prob": round(candidate_vegas_p, 4),
                            "edge": round(candidate_edge, 4),
                            "is_favorite": is_favorite,
                            "odds_bucket": pick["odds_bucket"],
                            "season_phase": pick["season_phase"],
                            "edge_threshold": pick["edge_threshold"],
                            "policy_reason": pick["policy_reason"],
                            "policy_name": policy.name,
                            "won": bool(candidate_won),
                            "game_pnl": 0,
                            "bankroll": bankroll}
            records.append(enrich_bet_record(
                rec, row=row, side=candidate_side, model_prob=candidate_p,
                vegas_prob=candidate_vegas_p, odds=candidate_ml,
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

        rec = {**base,
            "candidate_side": bet_side,
            "bet_side":   bet_side,
            "stake":      stake,
            "odds":       pick_ml,
            "model_prob": round(bet_p, 4),
            "vegas_prob": round(vegas_p, 4),
            "edge":       round(bet_edge, 4),
            "is_favorite": bool(pick_ml < 0),
            "odds_bucket": pick["odds_bucket"],
            "season_phase": pick["season_phase"],
            "edge_threshold": pick["edge_threshold"],
            "policy_reason": pick["policy_reason"],
            "policy_name": policy.name,
            "won":        won,
            "game_pnl":   game_pnl,
            "bankroll":   bankroll,
        }
        records.append(enrich_bet_record(
            rec, row=row, side=bet_side, model_prob=bet_p,
            vegas_prob=vegas_p, odds=pick_ml,
        ))

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, year: int) -> dict:
    bets = df[df["stake"] > 0]
    n    = len(bets)
    if n == 0:
        final = df["bankroll"].iloc[-1] if not df.empty else STARTING_BANKROLL
        return {
            "year": year, "games": len(df), "bets": 0, "wins": 0, "losses": 0,
            "win_pct": 0.0, "avg_edge": 0.0, "staked": 0.0, "pnl": 0.0,
            "roi": 0.0, "final_bk": round(final, 2),
        }
    wins   = int(bets["won"].sum())
    staked = bets["stake"].sum()
    pnl    = bets["game_pnl"].sum()
    roi    = pnl / staked * 100 if staked else 0
    final  = df["bankroll"].iloc[-1] if not df.empty else STARTING_BANKROLL
    return {
        "year":       year,
        "games":      len(df),
        "bets":       n,
        "wins":       wins,
        "losses":     n - wins,
        "win_pct":    round(wins / n * 100, 1),
        "avg_edge":   round(bets["edge"].mean() * 100, 1),
        "staked":     round(staked, 2),
        "pnl":        round(pnl, 2),
        "roi":        round(roi, 1),
        "final_bk":   round(final, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading features…")
    features = pd.read_csv(FEAT_PATH, parse_dates=["Date"])

    # Determine feature columns (same logic as train_model.py)
    import joblib
    bundle    = joblib.load(os.path.join(os.path.dirname(__file__), "models", "win_prob_model.pkl"))
    feat_cols = bundle["features"]
    print(f"Feature cols: {len(feat_cols)} (market-independent)")

    all_records = []
    summaries   = []
    policy_rows = []

    for test_year in TEST_YEARS:
        train_df = features[features["year"] < test_year].copy()
        test_df  = features[features["year"] == test_year].copy()

        print(f"\n── {test_year} ─────────────────────────────────────────")
        print(f"  Train: {len(train_df):,} games ({features[features['year']<test_year]['year'].min()}–{test_year-1})")
        print(f"  Test : {len(test_df):,} games")

        print("  Loading odds…", end=" ")
        odds = load_odds(test_year)
        print(f"{len(odds)} valid rows")

        print("  Training model…", end=" ", flush=True)
        model = train_fold(train_df, feat_cols)
        print("done")

        policy, tuned = tune_policy(all_records, test_year)
        policy_rows.append(tuned)
        rule_text = ", ".join(
            f"{r.season_phase}:{r.bucket}>{r.min_edge:.0%}" for r in policy.rules
        ) or "no bet rules"
        print(f"  Policy: {policy.name} ({rule_text})")

        print("  Running Kelly simulation…", end=" ", flush=True)
        year_df = run_year(test_df, odds, model, feat_cols, policy)
        print("done")

        year_df["year"] = test_year
        all_records.append(year_df)

        s = summarize(year_df, test_year)
        summaries.append(s)
        print(f"  Bets: {s['bets']}  W/L: {s['wins']}-{s['losses']}  "
              f"Win%: {s['win_pct']}%  ROI: {s['roi']:+.1f}%  "
              f"P&L: ${s['pnl']:+.2f}  BK: ${s['final_bk']:.2f}")

    # Aggregate summary
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD SUMMARY  (5 out-of-sample seasons)")
    print(f"{'='*60}")
    print(f"  {'Year':<6} {'Bets':>5} {'W–L':>10} {'Win%':>6} {'Avg Edge':>9} {'ROI':>7} {'P&L':>9} {'Final BK':>10}")
    print(f"  {'─'*58}")

    total_bets = total_wins = total_staked = total_pnl = 0
    for s in summaries:
        wl = f"{s['wins']}–{s['losses']}"
        print(f"  {s['year']:<6} {s['bets']:>5} {wl:>10} {s['win_pct']:>5.1f}% "
              f"{s['avg_edge']:>8.1f}% {s['roi']:>+6.1f}% "
              f"${s['pnl']:>+8.2f} ${s['final_bk']:>9.2f}")
        total_bets   += s["bets"]
        total_wins   += s["wins"]
        total_staked += s["staked"]
        total_pnl    += s["pnl"]

    total_losses = total_bets - total_wins
    total_roi    = total_pnl / total_staked * 100 if total_staked else 0
    print(f"  {'─'*58}")
    print(f"  {'TOTAL':<6} {total_bets:>5} {total_wins}–{total_losses:>} "
          f"{total_wins/total_bets*100:>5.1f}%"
          f"           {total_roi:>+6.1f}% ${total_pnl:>+8.2f}")
    print(f"{'='*60}")

    # Compound across years (each year restarts at $100 for comparability)
    all_df = pd.concat(all_records, ignore_index=True)
    out = os.path.join(DATA_DIR, "walkforward_results.csv")
    all_df.to_csv(out, index=False)
    print(f"\n  Full results saved → {out}")

    if policy_rows:
        current_policy, current_tuning = tune_policy(all_records, max(TEST_YEARS) + 1)
        policy_rows.append(current_tuning)
        policy_df = pd.concat(policy_rows, ignore_index=True)
        policy_out = os.path.join(DATA_DIR, "walkforward_policy_rules.csv")
        policy_df.to_csv(policy_out, index=False)
        print(f"  Policy tuning report saved → {policy_out}")
        policy_artifact = os.path.join(MODELS_DIR, "betting_policy.json")
        save_policy(current_policy, policy_artifact)
        current_rule_text = ", ".join(
            f"{r.season_phase}:{r.bucket}>{r.min_edge:.0%}" for r in current_policy.rules
        ) or "no bet rules"
        print(f"  Current betting policy saved → {policy_artifact}")
        print(f"  Current policy: {current_policy.name} ({current_rule_text})")
