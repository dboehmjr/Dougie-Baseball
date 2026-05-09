"""
Fetch team batting splits (vs LHP / vs RHP) from the MLB Stats API.

Output: data/team_splits.csv
  team, year, ops_vs_lhp, ops_vs_rhp, pa_vs_lhp, pa_vs_rhp

Usage:
  python fetch_splits.py [start_year [end_year]]
"""

from __future__ import annotations

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import date

warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "team_splits.csv")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# MLB Stats API team IDs
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


def _get_team_split_ops(team_id: int, year: int, split: str) -> dict:
    """
    Fetch one team's batting splits for a given year and split type.
    split: 'vl' (vs LHP) or 'vr' (vs RHP)
    Returns dict: {ops, pa} or {ops: nan, pa: 0} on failure.
    """
    url = (f"{MLB_API_BASE}/teams/{team_id}/stats"
           f"?stats=statSplits&group=hitting&season={year}"
           f"&sitCodes={split}&sportId=1")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {"ops": np.nan, "pa": 0}
        s = splits[0].get("stat", {})
        ops = float(s.get("ops", np.nan) or np.nan)
        pa  = int(s.get("plateAppearances", 0) or 0)
        return {"ops": ops, "pa": pa}
    except Exception:
        return {"ops": np.nan, "pa": 0}


def fetch_team_splits(team_id: int, year: int) -> dict:
    """Return splits dict for one team-year."""
    abbr = TEAM_IDS.get(team_id, str(team_id))
    vl = _get_team_split_ops(team_id, year, "vl")
    vr = _get_team_split_ops(team_id, year, "vr")
    return {
        "team":       abbr,
        "year":       year,
        "ops_vs_lhp": vl["ops"],
        "ops_vs_rhp": vr["ops"],
        "pa_vs_lhp":  vl["pa"],
        "pa_vs_rhp":  vr["pa"],
    }


def fetch_all_splits(start_year: int = 2015,
                     end_year: int | None = None,
                     cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Fetch splits for all teams for each year in [start_year, end_year].
    Incremental: only fetches years beyond the cached max.
    """
    if end_year is None:
        end_year = date.today().year

    existing = None
    fetch_years = list(range(start_year, end_year + 1))

    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        max_cached = int(existing["year"].max())
        # Always re-fetch current year (partial season)
        fetch_years = [y for y in fetch_years if y > max_cached or y == end_year]
        if not fetch_years:
            print(f"Team splits cache up to date (through {max_cached})")
            return existing
        print(f"Team splits cache through {max_cached}; fetching {fetch_years}…")

    rows = []
    for year in fetch_years:
        print(f"  Fetching team batting splits {year}…")
        for team_id in TEAM_IDS:
            row = fetch_team_splits(team_id, year)
            rows.append(row)
            time.sleep(0.05)   # be gentle — 30 teams × 2 splits each
        print(f"    Done ({year})")

    if not rows:
        return existing if existing is not None else pd.DataFrame()

    new_data = pd.DataFrame(rows)

    combined = (
        pd.concat([existing, new_data], ignore_index=True)
        if existing is not None else new_data
    )
    # Drop old rows for re-fetched years
    for yr in fetch_years:
        if existing is not None:
            combined = combined[~(
                (combined["year"] == yr) &
                (combined.index < len(existing))
            )]
    combined = (combined.sort_values(["year", "team"])
                         .drop_duplicates(["team", "year"])
                         .reset_index(drop=True))
    combined.to_csv(cache_path, index=False)
    print(f"  Saved {len(combined):,} rows → {cache_path}")
    return combined


def get_team_split_ops(team: str,
                        year: int,
                        sp_throws: str | None,
                        splits_df: pd.DataFrame) -> float:
    """
    Return the batting team's OPS against pitchers of hand sp_throws.
    sp_throws: 'L', 'R', or None
    Falls back to overall OPS average if split not found.
    """
    if splits_df is None or splits_df.empty or sp_throws is None:
        return np.nan

    row = splits_df[(splits_df["team"] == team) & (splits_df["year"] == year)]
    if row.empty:
        # Try prior year
        row = splits_df[(splits_df["team"] == team) & (splits_df["year"] == year - 1)]
    if row.empty:
        return np.nan

    r = row.iloc[0]
    if sp_throws == "L":
        return float(r.get("ops_vs_lhp", np.nan))
    elif sp_throws == "R":
        return float(r.get("ops_vs_rhp", np.nan))
    else:
        # Average of both splits (weighted by PA if available)
        pa_l = float(r.get("pa_vs_lhp", 0) or 0)
        pa_r = float(r.get("pa_vs_rhp", 0) or 0)
        ops_l = float(r.get("ops_vs_lhp", np.nan))
        ops_r = float(r.get("ops_vs_rhp", np.nan))
        total_pa = pa_l + pa_r
        if total_pa > 0 and not np.isnan(ops_l) and not np.isnan(ops_r):
            return (ops_l * pa_l + ops_r * pa_r) / total_pa
        return np.nanmean([ops_l, ops_r])


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    end   = int(sys.argv[2]) if len(sys.argv) > 2 else date.today().year

    df = fetch_all_splits(start_year=start, end_year=end)
    print(f"\nDone. {len(df):,} team-seasons in {CACHE_PATH}")
    print(df.head(10).to_string(index=False))
