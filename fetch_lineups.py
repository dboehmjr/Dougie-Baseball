"""
Fetch confirmed starting lineups from the MLB Stats API and compute
average lineup OPS for each team.

This is a prediction-time-only feature (lineups aren't known at training time).
The model handles NaN gracefully via SimpleImputer.

Output: in-memory dict (not cached to disk — lineups change daily)

Functions:
  fetch_confirmed_lineups(game_date)  → DataFrame of all games with lineup OPS
  get_lineup_ops(home_team, away_team, lineups_df)  → (home_ops, away_ops)
"""

from __future__ import annotations

import os
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime
from functools import lru_cache

warnings.filterwarnings("ignore")

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

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
ABBR_TO_ID = {v: k for k, v in TEAM_IDS.items()}


@lru_cache(maxsize=500)
def _get_player_ops(person_id: int, year: int) -> float:
    """Fetch a batter's OPS for the given season. Cached per player-year."""
    url = (f"{MLB_API_BASE}/people/{person_id}/stats"
           f"?stats=season&group=hitting&season={year}&sportId=1")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                ops = s.get("ops")
                if ops is not None:
                    return float(ops)
    except Exception:
        pass
    return np.nan


def _compute_lineup_avg_ops(player_ids: list[int], year: int) -> float:
    """Average OPS of the 9 lineup spots. NaN players are excluded from mean."""
    ops_vals = [_get_player_ops(pid, year) for pid in player_ids]
    valid = [v for v in ops_vals if not np.isnan(v)]
    return float(np.mean(valid)) if valid else np.nan


def _fetch_game_lineups(game_pk: int, year: int) -> dict:
    """
    Fetch confirmed lineups for a single game_pk.
    Returns: {'home_ops': float, 'away_ops': float,
               'home_sp_id': int, 'away_sp_id': int,
               'home_confirmed': bool, 'away_confirmed': bool}
    """
    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
    result = {
        "home_lineup_ops": np.nan, "away_lineup_ops": np.nan,
        "home_lineup_confirmed": False, "away_lineup_confirmed": False,
    }
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return result

    for side in ("home", "away"):
        team_data = data.get("teams", {}).get(side, {})
        batting_order = team_data.get("battingOrder", [])
        players       = team_data.get("players", {})

        if not batting_order:
            continue

        # battingOrder is a list of player IDs (as ints)
        lineup_ids = [int(pid) for pid in batting_order[:9]]
        if len(lineup_ids) >= 8:   # accept partial lineup (DH rules vary)
            result[f"{side}_lineup_confirmed"] = True
            result[f"{side}_lineup_ops"] = _compute_lineup_avg_ops(lineup_ids, year)

    return result


def fetch_confirmed_lineups(game_date: str | None = None) -> pd.DataFrame:
    """
    For all games on game_date, fetch confirmed lineup OPS.

    Returns DataFrame:
      game_date, home_team, away_team,
      home_lineup_ops, away_lineup_ops,
      home_lineup_confirmed, away_lineup_confirmed
    """
    if game_date is None:
        game_date = date.today().isoformat()

    year = int(game_date[:4])

    # Get the day's schedule
    url = f"{MLB_API_BASE}/schedule?sportId=1&date={game_date}&hydrate=teams"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        schedule = resp.json()
    except Exception as exc:
        print(f"  Lineup fetch: schedule request failed — {exc}")
        return pd.DataFrame()

    rows = []
    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            game_pk   = game["gamePk"]
            home_name = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_name = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            home_id   = game.get("teams", {}).get("home", {}).get("team", {}).get("id", 0)
            away_id   = game.get("teams", {}).get("away", {}).get("team", {}).get("id", 0)
            home_abbr = TEAM_IDS.get(home_id, home_name[:3].upper())
            away_abbr = TEAM_IDS.get(away_id, away_name[:3].upper())

            lineup_data = _fetch_game_lineups(game_pk, year)
            rows.append({
                "game_date":              game_date,
                "home_team":              home_abbr,
                "away_team":              away_abbr,
                "home_lineup_ops":        lineup_data["home_lineup_ops"],
                "away_lineup_ops":        lineup_data["away_lineup_ops"],
                "home_lineup_confirmed":  lineup_data["home_lineup_confirmed"],
                "away_lineup_confirmed":  lineup_data["away_lineup_confirmed"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    confirmed_pct = df["home_lineup_confirmed"].mean() * 100
    print(f"  Lineups: {len(df)} games, {confirmed_pct:.0f}% with confirmed home lineups")
    return df


def get_lineup_ops(home_team: str,
                   away_team: str,
                   lineups_df: pd.DataFrame) -> tuple[float, float]:
    """
    Return (home_lineup_ops, away_lineup_ops) for a matchup.
    Returns (nan, nan) if not found.
    """
    if lineups_df is None or lineups_df.empty:
        return np.nan, np.nan

    row = lineups_df[
        (lineups_df["home_team"] == home_team) &
        (lineups_df["away_team"] == away_team)
    ]
    if row.empty:
        return np.nan, np.nan

    r = row.iloc[0]
    return float(r.get("home_lineup_ops", np.nan)), float(r.get("away_lineup_ops", np.nan))


if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    print(f"Fetching confirmed lineups for {target_date}…")
    df = fetch_confirmed_lineups(game_date=target_date)

    if df.empty:
        print("No lineup data found.")
    else:
        print(f"\n{len(df)} games:\n")
        cols = ["home_team", "away_team", "home_lineup_ops",
                "away_lineup_ops", "home_lineup_confirmed"]
        print(df[cols].to_string(index=False))
