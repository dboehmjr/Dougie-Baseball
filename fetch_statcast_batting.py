"""
Fetch team-level Statcast batting metrics from Baseball Savant via pybaseball,
enriched with team assignments from the MLB Stats API.

Baseball Savant's exit-velocity leaderboard gives accurate raw counts (BBE,
barrels, hard-hit events, avg EV) per player, but omits the team column.
We resolve team assignments using the MLB Stats API's season player roster.
Players traded mid-season are assigned to their end-of-season team — a minor
approximation that is good enough for a signal feature.

Metrics collected (per team, per season):
  barrel_pct    — barrels / batted ball events  (quality of contact)
  hard_hit_pct  — batted balls with exit velo ≥ 95 mph / BBE
  avg_exit_velo — average exit velocity (mph)

Used as PRIOR-SEASON features (year Y-1 → games in year Y), matching the
same no-leakage pattern as home_team_ops.  Statcast data available 2015+.

Output: data/statcast_batting.csv
  team, year, barrel_pct, hard_hit_pct, avg_exit_velo

Usage:
  python fetch_statcast_batting.py
  python fetch_statcast_batting.py --start 2015 --end 2025
  python fetch_statcast_batting.py --refresh
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import pybaseball as pb
import requests

pb.cache.enable()

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "statcast_batting.csv")
os.makedirs(DATA_DIR, exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"

# Baseball Savant / MLB API abbreviation → our 3-letter code
ABBREV_MAP = {
    "WSH": "WSN", "SD":  "SDP", "TB":  "TBR",
    "KC":  "KCR", "SF":  "SFG", "CWS": "CHW",
    "AZ":  "ARI", "LAA": "LAA", "ATH": "OAK",
}


def _norm(abbrev: str) -> str:
    s = str(abbrev).strip().upper()
    return ABBREV_MAP.get(s, s)


def _get_team_abbrev_map(year: int) -> dict[int, str]:
    """Return {team_id: abbrev} from the MLB Stats API for the given season."""
    try:
        r = requests.get(f"{MLB_API}/teams",
                         params={"sportId": 1, "season": year}, timeout=15)
        r.raise_for_status()
        return {t["id"]: _norm(t["abbreviation"])
                for t in r.json().get("teams", []) if "abbreviation" in t}
    except Exception as exc:
        print(f"  WARNING: failed to fetch team list ({exc})")
        return {}


def _get_player_team_map(year: int, team_abbrev_map: dict[int, str]) -> dict[int, str]:
    """
    Return {mlbam_player_id: team_abbrev} from the MLB Stats API.
    Uses the season player list — reflects end-of-season team.
    """
    try:
        r = requests.get(f"{MLB_API}/sports/1/players",
                         params={"season": year}, timeout=20)
        r.raise_for_status()
        result = {}
        for p in r.json().get("people", []):
            pid = p.get("id")
            tid = (p.get("currentTeam") or {}).get("id")
            if pid and tid:
                abbrev = team_abbrev_map.get(tid)
                if abbrev:
                    result[pid] = abbrev
        return result
    except Exception as exc:
        print(f"  WARNING: failed to fetch player roster ({exc})")
        return {}


def _fetch_year(year: int) -> pd.DataFrame | None:
    """
    Fetch per-player Statcast exit velocity / barrel data for one season,
    resolve team assignments from the MLB Stats API, then aggregate to team totals.
    """
    print(f"  Fetching Statcast batting {year}...", end=" ", flush=True)

    # 1. Player-level Statcast data from Baseball Savant
    try:
        df = pb.statcast_batter_exitvelo_barrels(year, minBBE=0)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None

    if df is None or df.empty:
        print("no data returned")
        return None

    # 2. Column detection — Baseball Savant occasionally renames things
    id_col      = next((c for c in ["player_id"] if c in df.columns), None)
    bbe_col     = next((c for c in ["attempts", "bbe", "pa"] if c in df.columns), None)
    barrel_col  = next((c for c in ["barrels", "barrel"] if c in df.columns), None)
    hard_hit_col= next((c for c in ["ev95plus", "hard_hit"] if c in df.columns), None)
    ev_col      = next((c for c in ["avg_hit_speed", "launch_speed_avg",
                                     "avg_exit_velocity"] if c in df.columns), None)

    if not bbe_col or not barrel_col or not id_col:
        print(f"WARNING: expected columns missing. Available: {df.columns.tolist()}")
        return None

    # 3. Resolve team per player from MLB Stats API
    team_abbrev_map  = _get_team_abbrev_map(year)
    player_team_map  = _get_player_team_map(year, team_abbrev_map)

    df = df.copy()
    df["_team"] = df[id_col].map(player_team_map)
    # Drop rows with no team resolved (prospects, coaches, etc.)
    df = df[df["_team"].notna()]

    if df.empty:
        print("no team matches after roster join")
        return None

    df[bbe_col]    = pd.to_numeric(df[bbe_col],    errors="coerce").fillna(0)
    df[barrel_col] = pd.to_numeric(df[barrel_col], errors="coerce").fillna(0)

    agg: dict = {"_bbe": (bbe_col, "sum"), "_barrels": (barrel_col, "sum")}
    if hard_hit_col:
        df[hard_hit_col] = pd.to_numeric(df[hard_hit_col], errors="coerce").fillna(0)
        agg["_hh"] = (hard_hit_col, "sum")
    if ev_col:
        df[ev_col] = pd.to_numeric(df[ev_col], errors="coerce")
        agg["_ev"] = (ev_col, "mean")

    team_df = df.groupby("_team").agg(**agg).reset_index()
    team_df.rename(columns={"_team": "team"}, inplace=True)

    safe_div = lambda num, den: np.where(den > 0, num / den, np.nan)
    team_df["barrel_pct"]    = safe_div(team_df["_barrels"], team_df["_bbe"])
    team_df["hard_hit_pct"]  = safe_div(team_df["_hh"],     team_df["_bbe"]) \
                                if "_hh" in team_df.columns else np.nan
    team_df["avg_exit_velo"] = team_df["_ev"] if "_ev" in team_df.columns else np.nan

    keep = ["team", "barrel_pct", "hard_hit_pct", "avg_exit_velo"]
    team_df = team_df[[c for c in keep if c in team_df.columns]].copy()
    team_df["year"] = year

    n_teams = len(team_df)
    hh_cov  = int(team_df["hard_hit_pct"].notna().sum()) if "hard_hit_pct" in team_df.columns else 0
    print(f"{n_teams} teams  "
          f"(barrel: {team_df['barrel_pct'].notna().sum()}, hard-hit: {hh_cov})")
    return team_df


def fetch_statcast_batting(start_year: int = 2015,
                           end_year: int | None = None,
                           cache_path: str = CACHE_PATH,
                           refresh: bool = False) -> pd.DataFrame:
    """
    Fetch Statcast batting metrics for all seasons in [start_year, end_year].
    Incremental: only fetches years beyond the cached max.
    """
    if end_year is None:
        end_year = pd.Timestamp.today().year

    existing = None
    fetch_years = list(range(start_year, end_year + 1))

    if os.path.exists(cache_path) and not refresh:
        existing = pd.read_csv(cache_path)
        if existing.empty:
            existing = None
        else:
            max_cached = int(existing["year"].max())
            # Always re-fetch current year (partial season in progress)
            fetch_years = [y for y in fetch_years if y > max_cached or y == end_year]
            if not fetch_years:
                print(f"Statcast batting cache up to date (through {max_cached})")
                return existing
            print(f"Statcast batting cache through {max_cached}; fetching {fetch_years}…")

    frames = []
    for year in fetch_years:
        df = _fetch_year(year)
        if df is not None:
            frames.append(df)
        time.sleep(2.0)

    if not frames:
        print("  No new data fetched.")
        return existing if existing is not None else pd.DataFrame()

    new_data = pd.concat(frames, ignore_index=True)

    if existing is not None:
        combined = pd.concat(
            [existing[~existing["year"].isin(fetch_years)], new_data],
            ignore_index=True,
        )
    else:
        combined = new_data

    combined = (combined
                .sort_values(["year", "team"])
                .drop_duplicates(["team", "year"])
                .reset_index(drop=True))
    combined.to_csv(cache_path, index=False)
    print(f"  Saved {len(combined):,} team-seasons → {cache_path}")
    return combined


def get_team_statcast(team: str,
                      year: int,
                      statcast_df: pd.DataFrame) -> dict:
    """
    Return Statcast batting metrics for a team-year.
    Falls back to league median for that year if team not found.
    """
    empty = {"barrel_pct": np.nan, "hard_hit_pct": np.nan, "avg_exit_velo": np.nan}
    if statcast_df is None or statcast_df.empty:
        return empty

    row = statcast_df[(statcast_df["team"] == team) & (statcast_df["year"] == year)]
    if not row.empty:
        r = row.iloc[0]
        return {k: float(r.get(k, np.nan)) if k in r.index else np.nan for k in empty}

    yr_df = statcast_df[statcast_df["year"] == year]
    if yr_df.empty:
        return empty
    return {k: float(yr_df[k].median()) if k in yr_df.columns else np.nan for k in empty}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Statcast team batting metrics")
    parser.add_argument("--start",   type=int, default=2015)
    parser.add_argument("--end",     type=int, default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if cache exists")
    args = parser.parse_args()

    df = fetch_statcast_batting(args.start, args.end, refresh=args.refresh)

    if not df.empty:
        yr = min(2024, df["year"].max())
        print(f"\nSample (all teams, {yr}):")
        sample = df[df["year"] == yr].sort_values("barrel_pct", ascending=False)
        print(sample[["team", "year", "barrel_pct", "hard_hit_pct",
                       "avg_exit_velo"]].head(10).to_string(index=False))
