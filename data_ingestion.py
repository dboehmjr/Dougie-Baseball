"""
Pull and cache MLB game-level data using pybaseball.
Produces a flat CSV of one row per game with features for a pre-game
win probability model.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import pybaseball as pb

pb.cache.enable()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_YEARS = list(range(2015, 2026))   # 2015–2025
TEST_YEARS  = [2026]                     # current season (partial)


# ---------------------------------------------------------------------------
# Schedule / results
# ---------------------------------------------------------------------------

def fetch_schedule(year: int) -> pd.DataFrame:
    """Return completed regular-season games for a given year."""
    sched = pb.schedule_and_record(year, "NYY")  # placeholder team – we pull all below
    # pybaseball doesn't have a league-wide schedule endpoint; use standings loop
    # instead we pull team-level schedules for all 30 teams and deduplicate
    return sched


def fetch_all_game_logs(years: list[int], cache_path: str | None = None) -> pd.DataFrame:
    """
    Pull team game logs for every team + year.  Each row = one team's view of a
    game.  We join home vs. away below.

    Incremental: if a cache exists, only fetches years beyond the cached max.
    """
    existing = None
    fetch_years = list(years)

    if cache_path and os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, parse_dates=["Date"])
        if existing.empty:
            existing = None
        else:
            max_cached = int(existing["year"].max())
            fetch_years = [y for y in years if y > max_cached]
            if not fetch_years:
                print(f"Game log cache up to date (through {max_cached}); loading from {cache_path}")
                return existing
            print(f"Game log cache has through {max_cached}; fetching {fetch_years}...")

    teams = [
        "ARI","ATL","BAL","BOS","CHC","CHW","CIN","CLE","COL","DET",
        "HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYM","NYY","ATH",
        "PHI","PIT","SDP","SEA","SFG","STL","TBR","TEX","TOR","WSN",
    ]

    frames = []
    for year in fetch_years:
        print(f"Fetching {year}...")
        for team in teams:
            try:
                df = pb.schedule_and_record(year, team)
                df["team"] = team
                df["year"] = year
                frames.append(df)
                time.sleep(0.3)   # polite rate limiting
            except Exception as exc:
                print(f"  Warning: {team} {year} failed — {exc}")

    if not frames:
        return existing

    new_data = pd.concat(frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data

    if cache_path:
        combined.to_csv(cache_path, index=False)
        print(f"Saved to {cache_path}")

    return combined


# ---------------------------------------------------------------------------
# Pitcher stats (season-level FIP as pre-game signal)
# ---------------------------------------------------------------------------

def fetch_pitcher_stats(years: list[int], cache_path: str | None = None) -> pd.DataFrame:
    """Incremental: only fetches years beyond the cached max."""
    existing = None
    fetch_years = list(years)

    if cache_path and os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        if existing.empty:
            existing = None
        else:
            max_cached = int(existing["year"].max())
            fetch_years = [y for y in years if y > max_cached]
            if not fetch_years:
                print(f"Pitcher stats cache up to date (through {max_cached})")
                return existing
            print(f"Pitcher stats cache has through {max_cached}; fetching {fetch_years}...")

    frames = []
    for year in fetch_years:
        try:
            df = pb.pitching_stats_bref(year)
            df["year"] = year
            frames.append(df)
            time.sleep(0.5)
        except Exception as exc:
            print(f"  Warning: pitcher stats {year} failed — {exc}")

    if not frames:
        return existing

    new_data = pd.concat(frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
    if cache_path:
        combined.to_csv(cache_path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Team batting stats — individual player stats from BRef, aggregated by team
# We fetch one year before the first train year so prior-season OPS is
# available for every season in the model (2014 → used for 2015 games, etc.)
# ---------------------------------------------------------------------------

def fetch_batting_stats_bref(years: list[int], cache_path: str | None = None) -> pd.DataFrame:
    """Incremental: only fetches years beyond the cached max."""
    existing = None
    fetch_years = list(years)

    if cache_path and os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        if existing.empty:
            existing = None
        else:
            max_cached = int(existing["year"].max())
            fetch_years = [y for y in years if y > max_cached]
            if not fetch_years:
                print(f"Batting stats cache up to date (through {max_cached})")
                return existing
            print(f"Batting stats cache through {max_cached}; fetching {fetch_years}...")

    frames = []
    for year in fetch_years:
        try:
            df = pb.batting_stats_bref(year)
            df["year"] = year
            frames.append(df)
            time.sleep(0.5)
        except Exception as exc:
            print(f"  Warning: batting stats {year} failed — {exc}")

    if not frames:
        return existing

    new_data = pd.concat(frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
    if cache_path:
        combined.to_csv(cache_path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_years = sorted(set(TRAIN_YEARS + TEST_YEARS))

    print("=== Fetching game logs ===")
    game_logs = fetch_all_game_logs(
        all_years,
        cache_path=os.path.join(DATA_DIR, "game_logs_raw.csv"),
    )
    print(f"Game logs shape: {game_logs.shape}")

    print("\n=== Fetching pitcher stats ===")
    # Start one year before the feature window so prior-season SP baselines
    # are available for the first modeled season.
    pitcher_years = sorted(set([min(all_years) - 1] + all_years))
    pitcher_stats = fetch_pitcher_stats(
        pitcher_years,
        cache_path=os.path.join(DATA_DIR, "pitcher_stats.csv"),
    )
    print(f"Pitcher stats shape: {pitcher_stats.shape}")

    print("\n=== Fetching batting stats (for team OPS) ===")
    # Start from one year before train start so prior-year OPS covers all seasons
    batting_years = sorted(set([min(all_years) - 1] + all_years))
    batting_stats = fetch_batting_stats_bref(
        batting_years,
        cache_path=os.path.join(DATA_DIR, "batting_stats.csv"),
    )
    print(f"Batting stats shape: {batting_stats.shape}")

    print("\nDone. Run feature_engineering.py next.")
