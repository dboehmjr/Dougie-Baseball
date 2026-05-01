"""
Fetch MLB moneylines from The Odds API and compute devigged home-win probabilities.

Requires env var: ODDS_API_KEY  (free tier at the-odds-api.com)

Output: data/odds_cache.csv (one row per game per day)
  game_date, home_team, away_team, home_ml, away_ml, home_implied, consensus_prob

Usage:
  python fetch_odds.py [YYYY-MM-DD]    # defaults to today
"""

from __future__ import annotations

import os
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime, timezone
import zoneinfo

warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "odds_cache.csv")

# Load from env var, .env file, or Streamlit secrets
def _load_env_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if key:
        return key
    # Try Streamlit secrets (works on Streamlit Community Cloud)
    try:
        import streamlit as st
        key = st.secrets.get("ODDS_API_KEY", "")
        if key:
            os.environ["ODDS_API_KEY"] = key  # cache for submodules
            return key
    except Exception:
        pass
    # Fall back to .env file (local dev)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ODDS_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""

ODDS_API_KEY = _load_env_key()
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

# The Odds API team name → our abbreviation
ODDS_TEAM_MAP = {
    "Arizona Diamondbacks":      "ARI",
    "Atlanta Braves":            "ATL",
    "Baltimore Orioles":         "BAL",
    "Boston Red Sox":            "BOS",
    "Chicago Cubs":              "CHC",
    "Chicago White Sox":         "CHW",
    "Cincinnati Reds":           "CIN",
    "Cleveland Guardians":       "CLE",
    "Colorado Rockies":          "COL",
    "Detroit Tigers":            "DET",
    "Houston Astros":            "HOU",
    "Kansas City Royals":        "KCR",
    "Los Angeles Angels":        "LAA",
    "Los Angeles Dodgers":       "LAD",
    "Miami Marlins":             "MIA",
    "Milwaukee Brewers":         "MIL",
    "Minnesota Twins":           "MIN",
    "New York Mets":             "NYM",
    "New York Yankees":          "NYY",
    "Oakland Athletics":         "OAK",
    "Philadelphia Phillies":     "PHI",
    "Pittsburgh Pirates":        "PIT",
    "San Diego Padres":          "SDP",
    "Seattle Mariners":          "SEA",
    "San Francisco Giants":      "SFG",
    "St. Louis Cardinals":       "STL",
    "Tampa Bay Rays":            "TBR",
    "Texas Rangers":             "TEX",
    "Toronto Blue Jays":         "TOR",
    "Washington Nationals":      "WSN",
    # Aliases
    "Cleveland Indians":         "CLE",
    "Anaheim Angels":            "LAA",
    "Sacramento River Cats":     "OAK",
    "Athletics":                 "OAK",
}


def american_to_implied(ml: float) -> float:
    """Convert American moneyline to raw implied probability (includes vig)."""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def devig_prob(home_ml: float, away_ml: float) -> float:
    """
    Remove the bookmaker's vig and return the devigged home-win probability.
    Uses the multiplicative / normalization method.
    """
    p_home = american_to_implied(home_ml)
    p_away = american_to_implied(away_ml)
    total  = p_home + p_away
    if total <= 0:
        return np.nan
    return p_home / total


def _normalize_team(name: str) -> str:
    return ODDS_TEAM_MAP.get(name, name.upper()[:3])


def fetch_mlb_odds(api_key: str = ODDS_API_KEY,
                   game_date: str | None = None,
                   bookmakers: str = "draftkings,fanduel,betmgm") -> pd.DataFrame:
    """
    Pull MLB moneylines from The Odds API for the given date.

    Returns DataFrame with columns:
      game_date, home_team, away_team, home_ml, away_ml,
      home_implied, away_implied, consensus_prob
    """
    if not api_key:
        print("  ODDS_API_KEY not set — skipping Vegas lines fetch")
        return pd.DataFrame()

    if game_date is None:
        game_date = date.today().isoformat()

    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h",
        "oddsFormat": "american",
        "bookmakers": bookmakers,
    }

    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  Odds API: {len(data)} games fetched | {remaining} requests remaining")
    except requests.RequestException as exc:
        print(f"  Odds API request failed: {exc}")
        return pd.DataFrame()

    rows = []
    for game in data:
        # Filter to the target date (API returns next 24-48 hours by default)
        commence = game.get("commence_time", "")
        try:
            game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            # Always classify game dates in Eastern time — avoids west-coast
            # games appearing on the wrong date when run on a Pacific machine.
            game_local_date = game_dt.astimezone(
                zoneinfo.ZoneInfo("America/New_York")
            ).date().isoformat()
        except Exception:
            game_local_date = game_date  # fallback

        home_raw  = game.get("home_team", "")
        away_raw  = game.get("away_team", "")
        home_abbr = _normalize_team(home_raw)
        away_abbr = _normalize_team(away_raw)

        # Collect lines from multiple books, then average
        # Also track DraftKings separately as a fallback for corrupt averages
        home_mls, away_mls = [], []
        dk_home_ml = dk_away_ml = None
        for book in game.get("bookmakers", []):
            is_dk = book.get("key") == "draftkings"
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team = _normalize_team(outcome["name"])
                    price = float(outcome["price"])
                    if team == home_abbr:
                        home_mls.append(price)
                        if is_dk:
                            dk_home_ml = price
                    elif team == away_abbr:
                        away_mls.append(price)
                        if is_dk:
                            dk_away_ml = price

        if not home_mls or not away_mls:
            continue

        home_ml_avg = float(np.mean(home_mls))
        away_ml_avg = float(np.mean(away_mls))

        # If either average falls in the impossible zone (-100, +100),
        # the consensus was corrupted by run-line odds — fall back to DraftKings
        def _in_bad_zone(ml): return -100 < ml < 100
        if (_in_bad_zone(home_ml_avg) or _in_bad_zone(away_ml_avg)) \
                and dk_home_ml is not None and dk_away_ml is not None:
            home_ml_avg = dk_home_ml
            away_ml_avg = dk_away_ml

        rows.append({
            "game_date":      game_local_date,
            "home_team":      home_abbr,
            "away_team":      away_abbr,
            "home_ml":        round(home_ml_avg, 1),
            "away_ml":        round(away_ml_avg, 1),
            "home_implied":   round(american_to_implied(home_ml_avg), 4),
            "away_implied":   round(american_to_implied(away_ml_avg), 4),
            "consensus_prob": round(devig_prob(home_ml_avg, away_ml_avg), 4),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Filter to the requested date
    df = df[df["game_date"] == game_date].reset_index(drop=True)
    return df


def load_or_fetch_odds(game_date: str | None = None,
                       api_key: str = ODDS_API_KEY,
                       cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Return odds for game_date from the cache if already fetched today;
    otherwise hit the API, append to cache, and return.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    # Read existing cache
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, dtype={"game_date": str})
        day_rows = cached[cached["game_date"] == game_date]
        if not day_rows.empty:
            return day_rows.reset_index(drop=True)
    else:
        cached = pd.DataFrame()

    # Fetch from API
    fresh = fetch_mlb_odds(api_key=api_key, game_date=game_date)
    if fresh.empty:
        return fresh

    # Append and save
    combined = (pd.concat([cached, fresh], ignore_index=True)
                  .drop_duplicates(["game_date", "home_team", "away_team"])
                  .sort_values(["game_date", "home_team"])
                  .reset_index(drop=True))
    combined.to_csv(cache_path, index=False)
    print(f"  Saved {len(combined):,} rows → {cache_path}")
    return fresh


def get_home_implied_prob(home_team: str,
                          away_team: str,
                          odds_df: pd.DataFrame) -> float:
    """
    Look up the devigged home-win probability for a matchup.
    Returns NaN if not found (model falls back to its own estimate).
    """
    if odds_df is None or odds_df.empty:
        return np.nan
    row = odds_df[
        (odds_df["home_team"] == home_team) &
        (odds_df["away_team"] == away_team)
    ]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["consensus_prob"])


def get_moneyline_str(home_team: str,
                      away_team: str,
                      odds_df: pd.DataFrame) -> str:
    """Return formatted moneyline string like '+140 / -165' for display."""
    if odds_df is None or odds_df.empty:
        return "N/A"
    row = odds_df[
        (odds_df["home_team"] == home_team) &
        (odds_df["away_team"] == away_team)
    ]
    if row.empty:
        return "N/A"
    r = row.iloc[0]
    def fmt(ml):
        return f"+{int(ml)}" if ml > 0 else str(int(ml))
    return f"{fmt(r['away_ml'])} / {fmt(r['home_ml'])}"


if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    if not ODDS_API_KEY:
        print("Set ODDS_API_KEY environment variable to use this script.")
        print("Free tier: https://the-odds-api.com  (500 requests/month)")
        sys.exit(0)

    print(f"Fetching MLB odds for {target_date}…")
    df = load_or_fetch_odds(game_date=target_date)
    if df.empty:
        print("No odds found.")
    else:
        print(f"\n{len(df)} games with odds:\n")
        for _, row in df.iterrows():
            ml_str = get_moneyline_str(row["home_team"], row["away_team"], df)
            print(f"  {row['away_team']:>3} @ {row['home_team']:<3}  ML: {ml_str:<14}  "
                  f"Vegas home prob: {row['consensus_prob']:.1%}")
