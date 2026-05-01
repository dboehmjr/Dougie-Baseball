"""
Fetch individual player WAR (Baseball Reference) for batters and pitchers.

Uses pybaseball's bwar_bat / bwar_pitch — no API key required.
Prior-year WAR is used as the quality proxy at prediction time (same logic
as team OPS and pitcher stuff — avoids in-season small-sample noise).

Output: data/player_war.csv
Columns: name_norm, year, war
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd
import pybaseball as pb

pb.cache.enable()

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "player_war.csv")


def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name).strip())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def fetch_player_war(start_year: int = 2015,
                     end_year: int | None = None,
                     cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Download bWAR for all batters and pitchers from start_year to end_year.
    Merges both tables (some players appear in both — e.g. two-way players),
    summing WAR so each (name_norm, year) row is unique.
    Saves to cache_path and returns the DataFrame.
    """
    if end_year is None:
        end_year = pd.Timestamp.today().year

    print("Fetching bWAR batting data…")
    bat = pb.bwar_bat(return_all=False)
    bat = bat[bat["year_ID"].between(start_year, end_year)].copy()
    # Exclude rows where this player was acting as a pitcher (two-way edge case)
    if "pitcher" in bat.columns:
        bat = bat[bat["pitcher"].map({True: False, False: True, 1: False, 0: True}).fillna(True)]
    bat = bat[["name_common", "year_ID", "WAR"]].copy()

    print("Fetching bWAR pitching data…")
    pit = pb.bwar_pitch(return_all=False)
    pit = pit[pit["year_ID"].between(start_year, end_year)].copy()
    pit = pit[["name_common", "year_ID", "WAR"]].copy()

    combined = pd.concat([bat, pit], ignore_index=True)
    combined["name_norm"] = combined["name_common"].apply(_normalize_name)
    combined = combined.rename(columns={"year_ID": "year", "WAR": "war"})
    combined = combined[["name_norm", "year", "war"]]

    # Sum WAR for the same player in the same year (two-way players, mid-season trades)
    combined = (combined.groupby(["name_norm", "year"], as_index=False)["war"]
                        .sum()
                        .sort_values(["year", "name_norm"])
                        .reset_index(drop=True))

    combined.to_csv(cache_path, index=False)
    print(f"Saved {len(combined):,} player-seasons → {cache_path}")
    return combined


def get_player_war(name: str, year: int, war_df: pd.DataFrame) -> float:
    """
    Return prior-year WAR for a player.  Uses year-1 as the quality signal
    (same convention as team OPS and pitcher stuff — avoids in-season noise).
    Falls back to year-2 if year-1 is missing, then 0.0.
    """
    norm = _normalize_name(name)
    for lookup_year in [year - 1, year - 2]:
        mask = (war_df["name_norm"] == norm) & (war_df["year"] == lookup_year)
        if mask.any():
            return float(war_df.loc[mask, "war"].iloc[0])
    return 0.0


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2014  # fetch 2014 so 2015 games have prior-year data
    df = fetch_player_war(start_year=start)
    print(f"\nSample:")
    print(df[df["year"] >= 2024].head(10).to_string(index=False))
