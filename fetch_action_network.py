"""
Scrape Action Network scoreboard API for MLB game results + opening/closing moneylines.

Usage:
    python fetch_action_network.py                  # fetch all 2026 games through yesterday
    python fetch_action_network.py --date 2026-04-20  # through a specific date
    python fetch_action_network.py --full             # re-fetch entire 2026 season
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH     = os.path.join(DATA_DIR, "action_network_odds.csv")
SEASON_START = date(2026, 3, 18)

# Book IDs we care about (prefer closing lines from major books)
# 15 = DraftKings, 30 = FanDuel, 76 = BetMGM, 123 = Caesars, 69 = Pinnacle
BOOK_IDS = "15,30,76,123,69"

# Abbreviation map: Action Network abbr → model abbr
AN_MAP = {
    "WSH": "WSN", "SD":  "SDP", "TB":  "TBR",
    "KC":  "KCR", "AZ":  "ARI", "SF":  "SFG",
    "OAK": "ATH", "ATH": "ATH", "CWS": "CHW",
}

os.makedirs(DATA_DIR, exist_ok=True)


def abbrev(raw: str) -> str:
    return AN_MAP.get(raw.upper(), raw.upper())


def fetch_games_for_date(game_date: date) -> list[dict]:
    url = "https://api.actionnetwork.com/web/v1/scoreboard/mlb"
    params = {
        "period":  "game",
        "bookIds": BOOK_IDS,
        "date":    game_date.strftime("%Y%m%d"),
    }
    try:
        r = requests.get(url, params=params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"    WARNING: {game_date} fetch failed: {exc}")
        return []

    rows = []
    for g in data.get("games", []):
        status      = g.get("status", "")
        real_status = g.get("real_status", "")
        if status not in ("complete", "final") and real_status not in ("closed", "complete", "final"):
            continue

        # Map team id → abbr, then use away_team_id / home_team_id
        teams = g.get("teams", [])
        team_map = {t["id"]: t.get("abbr", "") for t in teams if "id" in t}
        away_raw = team_map.get(g.get("away_team_id"), "")
        home_raw = team_map.get(g.get("home_team_id"), "")

        home = abbrev(home_raw)
        away = abbrev(away_raw)
        if not home or not away:
            continue

        bs      = g.get("boxscore", {})
        h_score = bs.get("total_home_points") or g.get("total_home_points")
        a_score = bs.get("total_away_points") or g.get("total_away_points")
        if h_score is None or a_score is None:
            continue

        # Collect moneylines from all available books, take consensus
        home_mls, away_mls = [], []
        for odds in g.get("odds", []):
            ml_h = odds.get("ml_home")
            ml_a = odds.get("ml_away")
            if ml_h and ml_a and ml_h != 0 and ml_a != 0:
                home_mls.append(ml_h)
                away_mls.append(ml_a)

        home_ml = round(sum(home_mls) / len(home_mls)) if home_mls else None
        away_ml = round(sum(away_mls) / len(away_mls)) if away_mls else None

        rows.append({
            "date":       str(game_date),
            "year":       game_date.year,
            "home_team":  home,
            "away_team":  away,
            "home_score": int(h_score),
            "away_score": int(a_score),
            "home_win":   int(int(h_score) > int(a_score)),
            "home_ml":    home_ml,
            "away_ml":    away_ml,
        })

    return rows


def fetch_all(through: date, full: bool = False) -> pd.DataFrame:
    existing = pd.DataFrame()
    if os.path.exists(OUT_PATH) and not full:
        existing = pd.read_csv(OUT_PATH, parse_dates=["date"])

    already = set(
        zip(existing["date"].astype(str), existing["home_team"])
    ) if not existing.empty else set()

    if full or existing.empty:
        fetch_from = SEASON_START
    else:
        last = pd.to_datetime(existing["date"].max()).date()
        fetch_from = last + timedelta(days=1)

    if fetch_from > through:
        print(f"  Already up to date through {through}")
        return existing

    print(f"  Fetching {fetch_from} → {through}…")
    all_rows = []
    d = fetch_from
    while d <= through:
        rows = fetch_games_for_date(d)
        new  = [r for r in rows if (str(r["date"]), r["home_team"]) not in already]
        if new:
            print(f"    {d}  {len(new)} games")
        all_rows.extend(new)
        d += timedelta(days=1)
        time.sleep(0.2)

    if not all_rows:
        print("  No new games found.")
        return existing

    new_df   = pd.DataFrame(all_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined["date"] = pd.to_datetime(combined["date"])
    combined = (combined
                .sort_values(["date", "home_team"])
                .drop_duplicates(subset=["date", "home_team", "away_team"]))
    combined.to_csv(OUT_PATH, index=False)
    print(f"\n  Total games saved: {len(combined):,}")
    print(f"  Saved → {OUT_PATH}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Fetch through YYYY-MM-DD")
    parser.add_argument("--full", action="store_true", help="Re-fetch entire 2026 season")
    args = parser.parse_args()

    through = (date.fromisoformat(args.date) if args.date
               else date.today() - timedelta(days=1))

    print(f"\n{'='*50}")
    print(f"  Action Network MLB scraper — through {through}")
    print(f"{'='*50}\n")

    df = fetch_all(through, full=args.full)

    if not df.empty:
        print(f"\n  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"  Games      : {len(df):,}")
        print(f"  With ML    : {df['home_ml'].notna().sum():,}")
        print(f"  Home win % : {df['home_win'].mean():.1%}")
        print()
        print(df.tail(5).to_string(index=False))
