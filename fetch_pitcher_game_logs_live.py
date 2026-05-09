"""
Fetch 2026 (current-season) pitcher game logs from the MLB Stats API.

Produces two files that parallel the Retrosheet-sourced historical files:
  data/pitcher_game_logs_live.csv   — per-appearance stats (outs, runs)
  data/game_sp_live.csv             — starting pitcher assignments per game

Pitcher IDs are stored as "mlb_{playerId}" to distinguish them from
Retrosheet IDs and to allow a clean union in feature_engineering.py.

Usage:
    python fetch_pitcher_game_logs_live.py            # fetch all 2026 games
    python fetch_pitcher_game_logs_live.py --year 2026
    python fetch_pitcher_game_logs_live.py --from 2026-04-01
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
LOGS_PATH   = os.path.join(DATA_DIR, "pitcher_game_logs_live.csv")
SP_PATH     = os.path.join(DATA_DIR, "game_sp_live.csv")
API_BASE    = "https://statsapi.mlb.com/api/v1"

SEASON_START = date(2026, 3, 18)

# MLB Stats API team ID → our abbreviation
TEAM_IDS = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS",
    112: "CHC", 145: "CHW", 113: "CIN", 114: "CLE",
    115: "COL", 116: "DET", 117: "HOU", 118: "KCR",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL",
    142: "MIN", 121: "NYM", 147: "NYY", 133: "ATH",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA",
    137: "SFG", 138: "STL", 139: "TBR", 140: "TEX",
    141: "TOR", 120: "WSN",
}


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}


def _abbrev(team_id: int) -> str:
    return TEAM_IDS.get(team_id, f"UNK{team_id}")


def _pitcher_id(mlb_id: int) -> str:
    return f"mlb_{mlb_id}"


def fetch_completed_games(start: date, end: date) -> list[dict]:
    """
    Return list of completed game dicts: {gamePk, game_date, home_team, away_team}.
    Chunks into 30-day windows to avoid oversized API responses.
    """
    games = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=29), end)
        data = _get(f"{API_BASE}/schedule", {
            "sportId": 1,
            "startDate": str(chunk_start),
            "endDate":   str(chunk_end),
            "hydrate":   "team",
        })
        for day in data.get("dates", []):
            for g in day.get("games", []):
                if "Final" not in g.get("status", {}).get("detailedState", ""):
                    continue
                home_id = g["teams"]["home"]["team"]["id"]
                away_id = g["teams"]["away"]["team"]["id"]
                games.append({
                    "gamePk":     g["gamePk"],
                    "game_date":  day["date"],
                    "home_team":  _abbrev(home_id),
                    "away_team":  _abbrev(away_id),
                })
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.1)
    return games


def fetch_boxscore(game_pk: int) -> dict:
    return _get(f"{API_BASE}/game/{game_pk}/boxscore")


def parse_boxscore(game: dict, boxscore: dict) -> tuple[list[dict], dict | None]:
    """
    Returns (log_rows, sp_row).
    log_rows: one row per pitcher appearance
    sp_row:   game_sp-style row with home/away SP assignments
    """
    log_rows = []
    sp_row   = None

    home_sp_id = home_sp_name = away_sp_id = away_sp_name = None

    for side_key, side_label, team_side in [("home", "home", 0), ("away", "away", 1)]:
        side = boxscore.get("teams", {}).get(side_key, {})
        pitcher_ids = side.get("pitchers", [])
        players     = side.get("players", {})
        if not pitcher_ids:
            continue

        first_pitcher_id = pitcher_ids[0]

        for i, pid in enumerate(pitcher_ids):
            player = players.get(f"ID{pid}", {})
            name   = player.get("person", {}).get("fullName", "Unknown")
            stats  = player.get("stats", {}).get("pitching", {})

            outs = stats.get("outs")
            runs = stats.get("runs")
            if outs is None or runs is None:
                continue

            is_starter = (pid == first_pitcher_id)
            row = {
                "game_date":     game["game_date"],
                "game_id":       f"mlb_{game['gamePk']}",
                "home_team":     game["home_team"],
                "away_team":     game["away_team"],
                "pitcher_id":    _pitcher_id(pid),
                "pitcher_name":  name,
                "team_side":     team_side,
                "is_starter":    is_starter,
                "outs_recorded": int(outs),
                "runs_allowed":  int(runs),
            }
            log_rows.append(row)

            if is_starter:
                if side_label == "home":
                    home_sp_id   = _pitcher_id(pid)
                    home_sp_name = name
                else:
                    away_sp_id   = _pitcher_id(pid)
                    away_sp_name = name

    if home_sp_id and away_sp_id:
        sp_row = {
            "Date":          game["game_date"],
            "year":          int(game["game_date"][:4]),
            "home_team":     game["home_team"],
            "away_team":     game["away_team"],
            "home_sp_id":    home_sp_id,
            "home_sp_name":  home_sp_name,
            "away_sp_id":    away_sp_id,
            "away_sp_name":  away_sp_name,
        }

    return log_rows, sp_row


def load_existing(path: str, key_cols: list[str]) -> tuple[pd.DataFrame, set]:
    if not os.path.exists(path):
        return pd.DataFrame(), set()
    df = pd.read_csv(path)
    already = set(zip(*[df[c].astype(str) for c in key_cols]))
    return df, already


def fetch_and_save(start: date, end: date) -> tuple[int, int]:
    """Fetch from start..end, skip already-cached games. Returns (new_log_rows, new_sp_rows)."""
    log_df, log_already = load_existing(LOGS_PATH, ["game_id"])
    sp_df,  sp_already  = load_existing(SP_PATH,   ["Date", "home_team"])

    games = fetch_completed_games(start, end)
    print(f"  Found {len(games)} completed games {start}–{end}")

    new_logs = []
    new_sps  = []
    skipped  = 0

    for g in games:
        game_id  = f"mlb_{g['gamePk']}"
        sp_key   = (g["game_date"], g["home_team"])

        if (game_id,) in log_already and sp_key in sp_already:
            skipped += 1
            continue

        box = fetch_boxscore(g["gamePk"])
        if not box:
            continue

        log_rows, sp_row = parse_boxscore(g, box)
        new_logs.extend(log_rows)
        if sp_row:
            new_sps.append(sp_row)
        time.sleep(0.12)

    if new_logs:
        combined = pd.concat([log_df, pd.DataFrame(new_logs)], ignore_index=True) \
                   if not log_df.empty else pd.DataFrame(new_logs)
        combined["game_date"] = pd.to_datetime(combined["game_date"])
        combined = combined.sort_values(["game_date", "game_id"]).drop_duplicates(
            subset=["game_id", "pitcher_id"]
        )
        combined.to_csv(LOGS_PATH, index=False)

    if new_sps:
        combined_sp = pd.concat([sp_df, pd.DataFrame(new_sps)], ignore_index=True) \
                      if not sp_df.empty else pd.DataFrame(new_sps)
        combined_sp["Date"] = pd.to_datetime(combined_sp["Date"])
        combined_sp = combined_sp.sort_values(["Date", "home_team"]).drop_duplicates(
            subset=["Date", "home_team", "away_team"]
        )
        combined_sp.to_csv(SP_PATH, index=False)

    print(f"  Skipped (cached): {skipped} | New log rows: {len(new_logs)} | New SP rows: {len(new_sps)}")
    return len(new_logs), len(new_sps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch live pitcher game logs from MLB Stats API")
    parser.add_argument("--year", type=int, default=date.today().year,
                        help="Season year (default: current year)")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (overrides --year)")
    parser.add_argument("--through", default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    through = date.fromisoformat(args.through) if args.through else date.today() - timedelta(days=1)
    start   = date.fromisoformat(args.from_date) if args.from_date else SEASON_START

    print(f"\nFetching pitcher game logs {start} → {through}")
    n_logs, n_sp = fetch_and_save(start, through)

    # Summary
    if os.path.exists(LOGS_PATH):
        df = pd.read_csv(LOGS_PATH)
        starters = df[df["is_starter"] == True]
        print(f"\n  pitcher_game_logs_live.csv : {len(df):,} rows  ({starters['game_date'].min()} – {starters['game_date'].max()})")
    if os.path.exists(SP_PATH):
        sp = pd.read_csv(SP_PATH)
        print(f"  game_sp_live.csv           : {len(sp):,} rows")
