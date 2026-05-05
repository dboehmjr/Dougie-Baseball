"""
Backfill SP pitch stuff into features.csv using game_sp.csv + pitcher_stuff.csv.

For each game in features.csv:
  1. Look up the home/away SP name from game_sp.csv
  2. Normalize the name
  3. Look up their prior-season stuff stats from pitcher_stuff.csv
     (falls back to team median if no direct match)
  4. Write home_sp_fbv, away_sp_fbv, sp_fbv_diff, home_sp_swstr, etc.

Run after fetch_pitcher_stuff.py has populated data/pitcher_stuff.csv.
"""

from __future__ import annotations

import os
import unicodedata
import re
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEATURES_PATH  = os.path.join(DATA_DIR, "features.csv")
GAME_SP_PATH   = os.path.join(DATA_DIR, "game_sp.csv")
STUFF_PATH     = os.path.join(DATA_DIR, "pitcher_stuff.csv")


def _normalize_name(name: str) -> str:
    name = str(name).strip()
    nfkd = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    name = name.encode("ascii", "ignore").decode("ascii").lower()
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    return " ".join(name.split())


def _lookup_stuff(name_norm: str, team: str, year: int,
                  stuff_df: pd.DataFrame) -> dict:
    """Direct name match, then team median, then league median."""
    empty = {"FBv": np.nan, "SwStr_pct": np.nan, "K_pct": np.nan, "xFIP": np.nan}

    if stuff_df.empty:
        return empty

    # Direct match (name + year)
    row = stuff_df[(stuff_df["name_norm"] == name_norm) & (stuff_df["year"] == year)]
    if not row.empty:
        r = row.iloc[0]
        return {k: (float(r[k]) if k in r and pd.notna(r[k]) else np.nan) for k in empty}

    # Team median for same year
    team_rows = stuff_df[(stuff_df["team"] == team) & (stuff_df["year"] == year)]
    if not team_rows.empty:
        result = {}
        for col in empty:
            if col in team_rows.columns:
                vals = pd.to_numeric(team_rows[col], errors="coerce").dropna()
                result[col] = float(vals.median()) if not vals.empty else np.nan
            else:
                result[col] = np.nan
        return result

    return empty


def backfill(features_path: str = FEATURES_PATH,
             game_sp_path: str  = GAME_SP_PATH,
             stuff_path: str    = STUFF_PATH) -> None:

    print("Loading data files…")
    features_df = pd.read_csv(features_path, parse_dates=["Date"])
    game_sp_df  = pd.read_csv(game_sp_path,  parse_dates=["Date"])
    stuff_df    = pd.read_csv(stuff_path)

    if stuff_df.empty:
        print("ERROR: pitcher_stuff.csv is empty. Run fetch_pitcher_stuff.py first.")
        return

    # Normalize SP names in game_sp
    game_sp_df["home_sp_norm"] = game_sp_df["home_sp_name"].apply(_normalize_name)
    game_sp_df["away_sp_norm"] = game_sp_df["away_sp_name"].apply(_normalize_name)

    # Index features by (Date, home_team, away_team)
    features_df["Date"] = pd.to_datetime(features_df["Date"])
    game_sp_df["Date"]  = pd.to_datetime(game_sp_df["Date"])

    # Merge SP assignments into features
    merged = features_df.merge(
        game_sp_df[["Date", "home_team", "away_team", "home_sp_norm", "away_sp_norm"]],
        on=["Date", "home_team", "away_team"],
        how="left",
    )

    total   = len(merged)
    matched = merged["home_sp_norm"].notna().sum()
    print(f"Matched SP names for {matched:,}/{total:,} games ({matched/total:.1%})")

    # SP stuff columns to fill
    stuff_cols = {
        "home_sp_fbv":   "FBv",
        "home_sp_swstr": "SwStr_pct",
        "home_sp_k_pct": "K_pct",
        "home_sp_xfip":  "xFIP",
        "away_sp_fbv":   "FBv",
        "away_sp_swstr": "SwStr_pct",
        "away_sp_k_pct": "K_pct",
        "away_sp_xfip":  "xFIP",
    }
    for col in stuff_cols:
        features_df[col] = np.nan

    print("Backfilling SP stuff…")
    n_direct = 0
    n_median = 0
    n_miss   = 0

    for idx, row in merged.iterrows():
        year       = int(row["year"]) - 1
        home_team  = row["home_team"]
        away_team  = row["away_team"]
        h_norm     = row.get("home_sp_norm", "")
        a_norm     = row.get("away_sp_norm", "")

        if pd.isna(h_norm) or h_norm == "":
            n_miss += 1
            continue

        h_stuff = _lookup_stuff(h_norm, home_team, year, stuff_df)
        a_stuff = _lookup_stuff(a_norm, away_team, year, stuff_df)

        # Track match quality using home SP as proxy
        direct_match = not stuff_df[
            (stuff_df["name_norm"] == h_norm) & (stuff_df["year"] == year)
        ].empty
        if direct_match:
            n_direct += 1
        else:
            n_median += 1

        features_df.at[idx, "home_sp_fbv"]   = h_stuff["FBv"]
        features_df.at[idx, "home_sp_swstr"]  = h_stuff["SwStr_pct"]
        features_df.at[idx, "home_sp_k_pct"]  = h_stuff["K_pct"]
        features_df.at[idx, "home_sp_xfip"]   = h_stuff["xFIP"]
        features_df.at[idx, "away_sp_fbv"]    = a_stuff["FBv"]
        features_df.at[idx, "away_sp_swstr"]  = a_stuff["SwStr_pct"]
        features_df.at[idx, "away_sp_k_pct"]  = a_stuff["K_pct"]
        features_df.at[idx, "away_sp_xfip"]   = a_stuff["xFIP"]

    # Recompute diff columns
    features_df["sp_fbv_diff"]   = features_df["home_sp_fbv"]   - features_df["away_sp_fbv"]
    features_df["sp_swstr_diff"] = features_df["home_sp_swstr"] - features_df["away_sp_swstr"]
    features_df["sp_k_pct_diff"] = features_df["home_sp_k_pct"] - features_df["away_sp_k_pct"]
    features_df["sp_xfip_diff"]  = features_df["away_sp_xfip"]  - features_df["home_sp_xfip"]

    print(f"  Direct SP matches : {n_direct:,}")
    print(f"  Team median used  : {n_median:,}")
    print(f"  No SP data        : {n_miss:,}")

    filled = features_df["home_sp_fbv"].notna().sum()
    print(f"  Rows with FBv now : {filled:,}/{total:,} ({filled/total:.1%})")

    features_df.to_csv(features_path, index=False)
    print(f"\nSaved updated features.csv ({len(features_df):,} rows)")


if __name__ == "__main__":
    backfill()
