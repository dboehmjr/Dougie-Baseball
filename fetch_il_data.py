"""
Fetch historical IL (Injured List) counts and quality scores for all 30 MLB teams.

Strategy: monthly snapshots (1st of each month, April–October) for each
season.  The most recent snapshot is then joined forward to each game date.

Outputs:
  data/il_counts.csv   — raw IL player count per team per snapshot date
  data/il_quality.csv  — WAR-weighted IL score per team per snapshot date

Run once; subsequent calls load from cache.  Pass --refresh to re-fetch.

Usage:
  python fetch_il_data.py
  python fetch_il_data.py --refresh
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import time
import unicodedata
from datetime import date

import numpy as np
import pandas as pd
import statsapi

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "il_counts.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ── MLB Stats API team IDs → our 3-letter abbreviations ───────────────────────
STATSAPI_TO_ABBREV: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KCR", 119: "LAD", 120: "WSN", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG", 138: "STL",
    139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}

ALL_TEAM_IDS = list(STATSAPI_TO_ABBREV.keys())

# Season months to sample: April (4) through October (10)
SAMPLE_MONTHS = [4, 5, 6, 7, 8, 9, 10]


# ── Fetch ─────────────────────────────────────────────────────────────────────

# 40-man roster IL designations (excludes minor-league "Injured 7-Day" and similar)
_MAJOR_LEAGUE_IL = {"Injured 10-Day", "Injured 15-Day", "Injured 60-Day"}


def get_il_count(team_id: int, snapshot_date: str) -> int:
    """
    Return the number of 40-man roster players on the IL for a team on a given date.
    Counts only 10-day / 15-day / 60-day IL — these are active roster designations.
    Excludes minor-league 7-day IL (concussions, etc.) which are org-level only.
    snapshot_date format: 'YYYY-MM-DD'
    """
    mm_dd_yyyy = pd.Timestamp(snapshot_date).strftime("%m/%d/%Y")

    def _fetch():
        return statsapi.get(
            "team_roster",
            {"teamId": team_id, "rosterType": "fullRoster", "date": mm_dd_yyyy},
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            roster = future.result(timeout=30)
        return sum(
            1 for p in roster.get("roster", [])
            if p.get("status", {}).get("description", "") in _MAJOR_LEAGUE_IL
        )
    except Exception:
        return np.nan  # fail with NaN so callers can distinguish from a true 0


def fetch_all_il_counts(start_year: int = 2015,
                        end_year:   int = date.today().year,
                        cache_path: str = CACHE_PATH,
                        refresh:    bool = False) -> pd.DataFrame:
    """
    Fetch IL snapshots for all teams and seasons.
    Incremental: skips (year, month) combinations already in the cache.
    """
    existing: pd.DataFrame | None = None
    done_keys: set[tuple[int, int]] = set()   # (year, month) already cached

    if os.path.exists(cache_path) and not refresh:
        existing = pd.read_csv(cache_path, parse_dates=["date"])
        if not existing.empty:
            existing["_ym"] = existing["date"].apply(lambda d: (d.year, d.month))
            done_keys = set(map(tuple, existing[["_ym"]].applymap(lambda x: x).values.tolist()))
            # Fix: rebuild from date column
            done_keys = set(zip(existing["date"].dt.year, existing["date"].dt.month))
            existing = existing.drop(columns=["_ym"], errors="ignore")
            max_year = existing["date"].dt.year.max()
            if max_year >= end_year and (end_year, 10) in done_keys:
                print(f"IL cache up to date (through {max_year}); loading from {cache_path}")
                print(f"  {len(existing):,} rows")
                return existing
            print(f"IL cache has through {max_year}; fetching missing months...")

    today = date.today()

    # Load existing rows into a mutable list for incremental saves
    all_rows: list[dict] = []
    if existing is not None:
        all_rows = existing.to_dict("records")

    for year in range(start_year, end_year + 1):
        for month in SAMPLE_MONTHS:
            if (year, month) in done_keys:
                continue
            # Don't fetch future dates
            snap = date(year, month, 1)
            if snap > today:
                continue

            snap_str = snap.strftime("%Y-%m-%d")
            print(f"  {snap_str} ...", end=" ", flush=True)
            snap_rows: list[dict] = []
            for team_id in ALL_TEAM_IDS:
                count = get_il_count(team_id, snap_str)
                snap_rows.append({
                    "date":     snap_str,
                    "team":     STATSAPI_TO_ABBREV[team_id],
                    "il_count": count,
                })
                time.sleep(0.15)   # ~30 teams × 0.15s = ~4.5s per snapshot

            all_rows.extend(snap_rows)
            print(f"{len(snap_rows)} teams")

            # ── Save after every snapshot so crashes don't lose progress ──
            tmp_df = pd.DataFrame(all_rows)
            tmp_df["date"] = pd.to_datetime(tmp_df["date"])
            tmp_df = tmp_df.drop_duplicates(subset=["date", "team"]).sort_values(["team", "date"])
            tmp_df.to_csv(cache_path, index=False)

    combined = pd.read_csv(cache_path, parse_dates=["date"]) if os.path.exists(cache_path) else pd.DataFrame(all_rows)
    print(f"\nSaved {len(combined):,} IL snapshots to {cache_path}")
    return combined


# ── Live lookup for predict.py ─────────────────────────────────────────────────

def get_current_il_count(team_abbrev: str,
                          game_date: str | None = None) -> float:
    """
    Return live IL count for a team.  Looks up the team_id from our mapping,
    then calls the Stats API.  Returns NaN on failure (unknown, not 0).
    """
    reverse = {v: k for k, v in STATSAPI_TO_ABBREV.items()}
    team_id = reverse.get(team_abbrev)
    if team_id is None:
        return np.nan
    snap_date = game_date or str(date.today())
    return get_il_count(team_id, snap_date)


# ── IL player names + WAR-quality score ───────────────────────────────────────

def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name).strip())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def get_il_players(team_id: int, snapshot_date: str) -> list[str]:
    """
    Return normalized names of all 40-man roster players on the IL
    for a team on snapshot_date.  Same API call as get_il_count.
    """
    mm_dd_yyyy = pd.Timestamp(snapshot_date).strftime("%m/%d/%Y")

    def _fetch():
        return statsapi.get(
            "team_roster",
            {"teamId": team_id, "rosterType": "fullRoster", "date": mm_dd_yyyy},
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            roster = future.result(timeout=30)
        names = []
        for p in roster.get("roster", []):
            if p.get("status", {}).get("description", "") in _MAJOR_LEAGUE_IL:
                full_name = p.get("person", {}).get("fullName", "")
                if full_name:
                    names.append(_normalize_name(full_name))
        return names
    except Exception:
        return []


def compute_il_war_score(player_names: list[str], year: int,
                          war_df: pd.DataFrame) -> float:
    """
    Sum prior-year WAR for all players in player_names.
    Uses year-1 as the quality signal (avoids in-season small-sample noise).
    Players not found in war_df contribute 0.
    """
    if war_df is None or war_df.empty or not player_names:
        return 0.0
    total = 0.0
    prior = year - 1
    for name in player_names:
        mask = (war_df["name_norm"] == name) & (war_df["year"] == prior)
        if mask.any():
            total += float(war_df.loc[mask, "war"].iloc[0])
        else:
            # Try year-2 as fallback for players returning from long absences
            mask2 = (war_df["name_norm"] == name) & (war_df["year"] == prior - 1)
            if mask2.any():
                total += float(war_df.loc[mask2, "war"].iloc[0])
    return round(total, 3)


QUALITY_CACHE_PATH = os.path.join(DATA_DIR, "il_quality.csv")


def fetch_all_il_quality(war_df: pd.DataFrame,
                          start_year: int = 2015,
                          end_year: int | None = None,
                          cache_path: str = QUALITY_CACHE_PATH,
                          refresh: bool = False) -> pd.DataFrame:
    """
    For every monthly snapshot already in il_counts.csv, fetch the list of
    IL players and compute their prior-year WAR sum.
    Incremental: skips (year, month) combinations already cached.

    Output: data/il_quality.csv  columns: date, team, il_war_score
    """
    if end_year is None:
        end_year = pd.Timestamp.today().year

    existing: pd.DataFrame | None = None
    done_keys: set[tuple[int, int]] = set()

    if os.path.exists(cache_path) and not refresh:
        existing = pd.read_csv(cache_path, parse_dates=["date"])
        if not existing.empty:
            done_keys = set(zip(existing["date"].dt.year, existing["date"].dt.month))
            max_year = existing["date"].dt.year.max()
            if max_year >= end_year and (end_year, 10) in done_keys:
                print(f"IL quality cache up to date (through {max_year})")
                return existing
            print(f"IL quality cache has through {max_year}; fetching missing…")

    today = date.today()
    all_rows: list[dict] = []
    if existing is not None:
        all_rows = existing.to_dict("records")

    for year in range(start_year, end_year + 1):
        for month in SAMPLE_MONTHS:
            if (year, month) in done_keys:
                continue
            snap = date(year, month, 1)
            if snap > today:
                continue
            snap_str = snap.strftime("%Y-%m-%d")
            print(f"  {snap_str} …", end=" ", flush=True)

            snap_rows: list[dict] = []
            for team_id, abbrev in STATSAPI_TO_ABBREV.items():
                players = get_il_players(team_id, snap_str)
                score   = compute_il_war_score(players, year, war_df)
                snap_rows.append({
                    "date":         snap_str,
                    "team":         abbrev,
                    "il_war_score": score,
                })
                time.sleep(0.15)

            all_rows.extend(snap_rows)
            print(f"{len(snap_rows)} teams")

            tmp_df = pd.DataFrame(all_rows)
            tmp_df["date"] = pd.to_datetime(tmp_df["date"])
            tmp_df = (tmp_df.drop_duplicates(subset=["date", "team"])
                            .sort_values(["team", "date"])
                            .reset_index(drop=True))
            tmp_df.to_csv(cache_path, index=False)

    combined = pd.read_csv(cache_path, parse_dates=["date"]) if os.path.exists(cache_path) else pd.DataFrame(all_rows)
    print(f"Saved {len(combined):,} IL quality rows → {cache_path}")
    return combined


def get_current_il_quality(team_abbrev: str, game_date: str,
                            war_df: pd.DataFrame) -> float:
    """
    Live IL WAR quality score for a team on game_date.
    Fetches IL player names from the MLB Stats API and sums prior-year WAR.
    """
    reverse = {v: k for k, v in STATSAPI_TO_ABBREV.items()}
    team_id = reverse.get(team_abbrev)
    if team_id is None:
        return 0.0
    year = pd.Timestamp(game_date).year
    players = get_il_players(team_id, game_date)
    return compute_il_war_score(players, year, war_df)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLB IL counts (monthly snapshots)")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if cache exists")
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end",   type=int, default=date.today().year)
    args = parser.parse_args()

    df = fetch_all_il_counts(args.start, args.end, refresh=args.refresh)
    print("\nSample (NYY):")
    sample = df[df["team"] == "NYY"].head(10)
    print(sample[["date", "team", "il_count"]].to_string(index=False))
