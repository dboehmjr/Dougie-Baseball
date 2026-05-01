"""
Scrape 2026 MLB moneylines from Action Network's internal API.

Fetches all games from season start through yesterday, averaging consensus
moneylines across all available books.  Saves to data/historical_odds.csv.

Usage:
  python fetch_historical_odds.py
  python fetch_historical_odds.py --start 2026-03-18 --end 2026-04-27
  python fetch_historical_odds.py --refresh   # re-fetch all dates
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "action_network_odds_2026.csv")
os.makedirs(DATA_DIR, exist_ok=True)

SEASON_START = date(2026, 3, 18)

API_URL = "https://api.actionnetwork.com/web/v1/scoreboard/mlb"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":   "application/json",
    "Origin":   "https://www.actionnetwork.com",
    "Referer":  "https://www.actionnetwork.com/mlb/odds",
}

# Action Network abbreviation → our 3-letter code
AN_ABBREV_MAP = {
    "KC":  "KCR",
    "SD":  "SDP",
    "TB":  "TBR",
    "SF":  "SFG",
    "WSH": "WSN",
    "CWS": "CHW",
    "AZ":  "ARI",
    "ATH": "OAK",
}


def _norm(abbr: str) -> str:
    s = str(abbr).strip().upper()
    return AN_ABBREV_MAP.get(s, s)


def american_to_implied(ml: float) -> float:
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)


def devig_prob(home_ml: float, away_ml: float) -> float:
    p_h = american_to_implied(home_ml)
    p_a = american_to_implied(away_ml)
    total = p_h + p_a
    return p_h / total if total > 0 else np.nan


def fetch_date(game_date: date) -> list[dict]:
    """Fetch all MLB games with odds for a single date from Action Network."""
    date_str = game_date.strftime("%Y%m%d")
    try:
        r = requests.get(API_URL, params={"period": "game", "date": date_str},
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        games = r.json().get("games", [])
    except Exception as exc:
        print(f"  {game_date}  ERROR: {exc}")
        return []

    rows = []
    for g in games:
        # Only completed games
        status = g.get("status", "")
        if "complete" not in status.lower() and "closed" not in status.lower():
            continue

        away_id   = g.get("away_team_id")
        home_id   = g.get("home_team_id")
        team_map  = {t["id"]: _norm(t["abbr"]) for t in g.get("teams", []) if "id" in t}
        home_abbr = team_map.get(home_id)
        away_abbr = team_map.get(away_id)
        if not home_abbr or not away_abbr:
            continue

        # Score
        bs       = g.get("boxscore", {}).get("stats", {})
        home_runs = bs.get("home", {}).get("runs")
        away_runs = bs.get("away", {}).get("runs")

        # Consensus ML: average across all books for type=game
        home_mls, away_mls = [], []
        for o in g.get("odds", []):
            if o.get("type") != "game":
                continue
            mh = o.get("ml_home")
            ma = o.get("ml_away")
            if mh is not None and ma is not None:
                try:
                    home_mls.append(float(mh))
                    away_mls.append(float(ma))
                except (TypeError, ValueError):
                    pass

        if not home_mls or not away_mls:
            continue

        home_ml_avg = float(np.mean(home_mls))
        away_ml_avg = float(np.mean(away_mls))

        rows.append({
            "game_date":      str(game_date),
            "home_team":      home_abbr,
            "away_team":      away_abbr,
            "home_ml":        round(home_ml_avg, 1),
            "away_ml":        round(away_ml_avg, 1),
            "home_implied":   round(american_to_implied(home_ml_avg), 4),
            "away_implied":   round(american_to_implied(away_ml_avg), 4),
            "consensus_prob": round(devig_prob(home_ml_avg, away_ml_avg), 4),
            "home_runs":      int(home_runs) if home_runs is not None else None,
            "away_runs":      int(away_runs) if away_runs is not None else None,
        })

    return rows


def fetch_historical_odds(
    start: date = SEASON_START,
    end: date | None = None,
    cache_path: str = CACHE_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    if end is None:
        end = date.today() - timedelta(days=1)  # only completed days

    existing = pd.DataFrame()
    fetch_dates: list[date] = []

    if os.path.exists(cache_path) and not refresh:
        existing = pd.read_csv(cache_path, dtype={"game_date": str})
        cached_dates = set(existing["game_date"].unique())
        all_dates    = [start + timedelta(days=i)
                        for i in range((end - start).days + 1)]
        fetch_dates  = [d for d in all_dates if str(d) not in cached_dates]
        if not fetch_dates:
            print(f"Historical odds cache up to date ({len(existing):,} rows).")
            return existing
        print(f"Cached {len(existing):,} rows. Fetching {len(fetch_dates)} missing dates…")
    else:
        fetch_dates = [start + timedelta(days=i)
                       for i in range((end - start).days + 1)]
        print(f"Fetching {len(fetch_dates)} dates ({start} → {end})…")

    all_rows = []
    for i, d in enumerate(fetch_dates):
        rows = fetch_date(d)
        n = len(rows)
        print(f"  {d}  {n} games", flush=True)
        all_rows.extend(rows)
        if i < len(fetch_dates) - 1:
            time.sleep(0.4)

    if not all_rows:
        print("No new data fetched.")
        return existing

    new_df = pd.DataFrame(all_rows)
    combined = (
        pd.concat([existing, new_df], ignore_index=True)
          .drop_duplicates(["game_date", "home_team", "away_team"])
          .sort_values(["game_date", "home_team"])
          .reset_index(drop=True)
    )
    combined.to_csv(cache_path, index=False)
    print(f"\nSaved {len(combined):,} rows → {cache_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch 2026 historical MLB odds from Action Network")
    parser.add_argument("--start",   default=str(SEASON_START))
    parser.add_argument("--end",     default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d   = date.fromisoformat(args.end) if args.end else None

    df = fetch_historical_odds(start=start_d, end=end_d, refresh=args.refresh)

    if not df.empty:
        print(f"\nSample (last 5 rows):")
        print(df.tail(5)[["game_date", "away_team", "home_team",
                           "home_ml", "away_ml", "consensus_prob"]].to_string(index=False))
        games_with_odds = len(df)
        print(f"\nTotal: {games_with_odds} games with odds data")
