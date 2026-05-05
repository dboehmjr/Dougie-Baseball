"""
Backfill MLB moneylines for 2022-2025 from Action Network's internal API.

Fetches completed games only, averages consensus moneylines across all
available books, and saves to data/action_network_odds_2022_2025.csv
in the same format as action_network_odds_2026.csv.

Usage:
  python fetch_odds_backfill.py               # fetch all of 2022-2025
  python fetch_odds_backfill.py --start 2024  # single season
  python fetch_odds_backfill.py --refresh     # re-fetch everything
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
CACHE_PATH = os.path.join(DATA_DIR, "action_network_odds_2022_2025.csv")
os.makedirs(DATA_DIR, exist_ok=True)

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

AN_ABBREV_MAP = {
    "KC":  "KCR", "SD":  "SDP", "TB":  "TBR", "SF":  "SFG",
    "WSH": "WSN", "CWS": "CHW", "AZ":  "ARI", "ATH": "OAK",
}

SEASON_DATES = {
    2022: (date(2022, 4, 7),  date(2022, 10, 5)),
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 20), date(2024, 9, 29)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
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


def _first_number(d: dict, names: list[str]) -> float | None:
    for name in names:
        val = d.get(name)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _avg(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def fetch_date(game_date: date) -> list[dict]:
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
        status = g.get("status", "")
        if "complete" not in status.lower() and "closed" not in status.lower():
            continue

        away_id  = g.get("away_team_id")
        home_id  = g.get("home_team_id")
        team_map = {t["id"]: _norm(t["abbr"]) for t in g.get("teams", []) if "id" in t}
        home_abbr = team_map.get(home_id)
        away_abbr = team_map.get(away_id)
        if not home_abbr or not away_abbr:
            continue

        bs        = g.get("boxscore", {}).get("stats", {})
        home_runs = bs.get("home", {}).get("runs")
        away_runs = bs.get("away", {}).get("runs")

        home_mls, away_mls = [], []
        open_home_mls, open_away_mls = [], []
        close_home_mls, close_away_mls = [], []
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
            oh = _first_number(o, ["ml_home_open", "open_ml_home", "opening_ml_home", "home_ml_open"])
            oa = _first_number(o, ["ml_away_open", "open_ml_away", "opening_ml_away", "away_ml_open"])
            ch = _first_number(o, ["ml_home_close", "close_ml_home", "closing_ml_home", "home_ml_close"])
            ca = _first_number(o, ["ml_away_close", "close_ml_away", "closing_ml_away", "away_ml_close"])
            if oh is not None and oa is not None:
                open_home_mls.append(oh)
                open_away_mls.append(oa)
            if ch is not None and ca is not None:
                close_home_mls.append(ch)
                close_away_mls.append(ca)

        if not home_mls or not away_mls:
            continue

        home_ml_avg = float(np.mean(home_mls))
        away_ml_avg = float(np.mean(away_mls))
        open_home_ml = _avg(open_home_mls)
        open_away_ml = _avg(open_away_mls)
        close_home_ml = _avg(close_home_mls)
        close_away_ml = _avg(close_away_mls)

        row = {
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
        }
        if open_home_ml is not None and open_away_ml is not None:
            row.update({
                "open_home_ml": round(open_home_ml, 1),
                "open_away_ml": round(open_away_ml, 1),
                "open_consensus_prob": round(devig_prob(open_home_ml, open_away_ml), 4),
            })
        if close_home_ml is not None and close_away_ml is not None:
            row.update({
                "close_home_ml": round(close_home_ml, 1),
                "close_away_ml": round(close_away_ml, 1),
                "close_consensus_prob": round(devig_prob(close_home_ml, close_away_ml), 4),
            })
        rows.append(row)

    return rows


def fetch_odds_backfill(seasons: list[int] | None = None,
                        cache_path: str = CACHE_PATH,
                        refresh: bool = False) -> pd.DataFrame:
    if seasons is None:
        seasons = list(SEASON_DATES.keys())

    existing = pd.DataFrame()
    cached_dates: set[str] = set()

    if os.path.exists(cache_path) and not refresh:
        existing = pd.read_csv(cache_path, dtype={"game_date": str})
        cached_dates = set(existing["game_date"].unique())
        print(f"Cache has {len(existing):,} rows ({len(cached_dates)} dates). Fetching missing...")

    all_rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    total_new = 0

    for season in seasons:
        start, end = SEASON_DATES[season]
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        missing = [d for d in dates if str(d) not in cached_dates]
        if not missing:
            print(f"  {season}: already cached ({len(dates)} dates)")
            continue
        print(f"  {season}: fetching {len(missing)} dates...")
        for i, d in enumerate(missing):
            rows = fetch_date(d)
            all_rows.extend(rows)
            total_new += len(rows)
            if (i + 1) % 30 == 0 or i == len(missing) - 1:
                print(f"    through {d} — {total_new} games so far")
                tmp = (pd.DataFrame(all_rows)
                         .drop_duplicates(["game_date", "home_team", "away_team"])
                         .sort_values(["game_date", "home_team"])
                         .reset_index(drop=True))
                tmp.to_csv(cache_path, index=False)
            if i < len(missing) - 1:
                time.sleep(0.4)

    combined = pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame(all_rows)
    print(f"\nDone. {len(combined):,} rows saved → {cache_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill 2022-2025 MLB odds from Action Network")
    parser.add_argument("--start",   type=int, default=2022, help="First season to fetch")
    parser.add_argument("--end",     type=int, default=2025, help="Last season to fetch")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    seasons = list(range(args.start, args.end + 1))
    df = fetch_odds_backfill(seasons=seasons, refresh=args.refresh)
    print(df.groupby(df["game_date"].str[:4])["game_date"].count().rename("games"))
