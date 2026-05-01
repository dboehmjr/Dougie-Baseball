"""
Download Retrosheet game logs and extract per-game starting pitcher assignments.
Retrosheet game logs are free public data at:
  https://www.retrosheet.org/gamelogs/gl{YEAR}.zip

Output: data/game_sp.csv
  Columns: date, home_team, away_team, home_sp_name, away_sp_name,
           home_sp_id, away_sp_id, year
"""

from __future__ import annotations

import io
import os
import time
import zipfile

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Retrosheet field positions (1-indexed → 0-indexed below)
COL_DATE         = 0    # Field 1
COL_AWAY_TEAM    = 3    # Field 4
COL_HOME_TEAM    = 6    # Field 7
COL_AWAY_SCORE   = 9    # Field 10
COL_HOME_SCORE   = 10   # Field 11
COL_AWAY_SP_ID   = 101  # Field 102
COL_AWAY_SP_NAME = 102  # Field 103
COL_HOME_SP_ID   = 103  # Field 104
COL_HOME_SP_NAME = 104  # Field 105

# Map Retrosheet 3-letter codes → our abbreviations
RETRO_TO_ABBREV = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHN": "CHC", "CHA": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCA": "KCR",
    "ANA": "LAA", "LAA": "LAA", "LAN": "LAD", "MIA": "MIA",
    "FLO": "MIA", "MIL": "MIL", "MIN": "MIN", "NYN": "NYM",
    "NYA": "NYY", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT",
    "SDN": "SDP", "SEA": "SEA", "SFN": "SFG", "SLN": "STL",
    "TBA": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
}


def download_year(year: int) -> pd.DataFrame | None:
    url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
    print(f"  Downloading {year}...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".TXT") or n.endswith(".txt")]
        if not name:
            print("no .TXT inside zip")
            return None
        with zf.open(name[0]) as f:
            lines = [line.decode("latin-1").strip() for line in f]

    rows = []
    for line in lines:
        parts = line.split(",")
        parts = [p.strip('"') for p in parts]
        if len(parts) < 105:
            continue
        rows.append({
            "date":          parts[COL_DATE],
            "away_team_raw": parts[COL_AWAY_TEAM],
            "home_team_raw": parts[COL_HOME_TEAM],
            "away_sp_id":    parts[COL_AWAY_SP_ID],
            "away_sp_name":  parts[COL_AWAY_SP_NAME],
            "home_sp_id":    parts[COL_HOME_SP_ID],
            "home_sp_name":  parts[COL_HOME_SP_NAME],
        })

    df = pd.DataFrame(rows)
    df["year"]      = year
    df["home_team"] = df["home_team_raw"].map(RETRO_TO_ABBREV).fillna(df["home_team_raw"])
    df["away_team"] = df["away_team_raw"].map(RETRO_TO_ABBREV).fillna(df["away_team_raw"])
    df["Date"]      = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])

    print(f"{len(df)} games")
    return df[["Date", "year", "home_team", "away_team",
               "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name"]]


def fetch_all_sp_data(years: list[int], cache_path: str) -> pd.DataFrame:
    """Incremental: only fetches years beyond the cached max."""
    existing = None
    fetch_years = list(years)

    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, parse_dates=["Date"])
        max_cached = int(existing["year"].max())
        fetch_years = [y for y in years if y > max_cached]
        if not fetch_years:
            print(f"SP cache up to date (through {max_cached}); loading from {cache_path}")
            return existing
        print(f"SP cache has through {max_cached}; fetching {fetch_years}...")

    frames = []
    for year in fetch_years:
        df = download_year(year)
        if df is not None:
            frames.append(df)
        time.sleep(1.0)

    if not frames:
        return existing

    new_data = pd.concat(frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
    combined.to_csv(cache_path, index=False)
    print(f"Saved SP data to {cache_path}")
    return combined


if __name__ == "__main__":
    # Retrosheet typically publishes the prior season by spring of the next year.
    # 2026 event files won't be available yet; graceful failure is handled inside download_year().
    years = list(range(2015, 2026))   # through 2025
    sp_data = fetch_all_sp_data(
        years,
        cache_path=os.path.join(DATA_DIR, "game_sp.csv"),
    )
    print(f"\nSP data shape: {sp_data.shape}")
    print(sp_data.head(3).to_string())
    print("\nDone. Run feature_engineering.py next.")
