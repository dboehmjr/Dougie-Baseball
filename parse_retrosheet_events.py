"""
Parse Retrosheet event files to produce per-game, per-pitcher statistics
AND per-game umpire data.

Downloads .EVA/.EVN event files from retrosheet.org/events/ for each year,
then parses them to extract:
  - Starting pitcher per team per game
  - Outs recorded per pitcher
  - Runs allowed per pitcher (not earned — earned run tracking requires
    full error/baserunner state; RA is a sufficient proxy for a model feature)
  - Home plate umpire per game (for umpire run-factor feature)

Outputs:
  data/pitcher_game_logs.csv
    game_date, game_id, home_team, away_team,
    pitcher_id, pitcher_name, team_side (0=away 1=home),
    is_starter, outs_recorded, runs_allowed

  data/umpire_game_logs.csv
    game_date, game_id, home_team, away_team, ump_name

Usage:
  python parse_retrosheet_events.py
"""

from __future__ import annotations

import io
import os
import re
import time
import zipfile

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Retrosheet → our abbreviations (same as fetch_sp_data.py)
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

# Primary event codes that record exactly 1 batter out
_ONE_OUT_PREFIXES = ("K", "G", "F", "L", "P", "!K")
# Events with 0 batter outs
_NO_OUT_PREFIXES  = ("S", "D", "T", "HR", "H", "W", "IW", "I",
                     "HP", "E", "FC", "SB", "BK", "WP", "PB",
                     "DI", "OA", "NP", "C")


# ---------------------------------------------------------------------------
# Event parsing helpers
# ---------------------------------------------------------------------------

def count_runs(event_str: str) -> int:
    """
    Count runs scored in a play.
    Runs = number of '-H' advance notations (dash = successful advance,
    X = out on bases → 'XH' is an out, not a run).
    """
    return event_str.count("-H")


def count_outs(event_str: str) -> int:
    """
    Count batter/baserunner outs in a single play record.
    Handles the most common cases; rare edge cases default to 0.
    """
    if not event_str or event_str == "NP":
        return 0

    # Strip advance notation (after first '.')
    primary = event_str.split(".")[0]
    # Strip modifier (after first '/')
    ev = primary.split("/")[0].strip()

    # Double/triple plays
    if "TP" in primary:
        return 3
    if "DP" in primary or "GDP" in primary or "LDP" in primary:
        return 2

    # Caught stealing / pickoff (baserunner out)
    if ev.startswith("CS") or (ev.startswith("PO") and "E" not in ev):
        return 1

    # Batter out
    if any(ev.startswith(p) for p in _ONE_OUT_PREFIXES):
        return 1

    # Fielding plays that end in a batter out: e.g. '43', '6-3', '5(2)4'
    if re.match(r"^\d", ev):
        return 1

    # No out events
    if any(ev.startswith(p) for p in _NO_OUT_PREFIXES):
        return 0

    return 0  # safe default for unrecognised events


# ---------------------------------------------------------------------------
# Game parser
# ---------------------------------------------------------------------------

def parse_event_file(lines: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Parse a list of lines from a Retrosheet .EVA or .EVN file.
    Returns (pitcher_records, ump_records):
      pitcher_records — one dict per pitcher-game appearance
      ump_records     — one dict per game with home plate umpire name
    """
    records     = []
    ump_records = []

    # State for current game
    game_id = game_date = vis_team = home_team = ump_name = None
    # pitcher[side] = (pitcher_id, pitcher_name, is_starter)
    pitcher: dict[int, tuple] = {}
    # cumulative stats per (side, pitcher_id): [outs, runs]
    stats: dict[tuple, list] = {}

    def flush_game():
        """Emit records for finished game."""
        if game_id is None:
            return
        ht = RETRO_TO_ABBREV.get(home_team, home_team)
        at = RETRO_TO_ABBREV.get(vis_team, vis_team)
        for (side, pid), (outs, runs) in stats.items():
            pname, is_starter = pitcher_meta.get((side, pid), ("unknown", False))
            records.append({
                "game_date":    game_date,
                "game_id":      game_id,
                "home_team":    ht,
                "away_team":    at,
                "pitcher_id":   pid,
                "pitcher_name": pname,
                "team_side":    side,       # 0=away, 1=home
                "is_starter":   is_starter,
                "outs_recorded": outs,
                "runs_allowed":  runs,
            })
        # Umpire record (one per game)
        if ump_name:
            ump_records.append({
                "game_date": game_date,
                "game_id":   game_id,
                "home_team": ht,
                "away_team": at,
                "ump_name":  ump_name,
            })

    pitcher_meta: dict[tuple, tuple] = {}  # (side, pid) → (name, is_starter)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        fields = line.split(",")
        rec = fields[0]

        if rec == "id":
            flush_game()
            game_id = fields[1] if len(fields) > 1 else None
            # Parse date from game_id: e.g. NYA20240405(0)
            if game_id and len(game_id) >= 11:
                try:
                    game_date = pd.Timestamp(
                        year=int(game_id[3:7]),
                        month=int(game_id[7:9]),
                        day=int(game_id[9:11]),
                    )
                except Exception:
                    game_date = None
            vis_team = home_team = ump_name = None
            pitcher = {}
            pitcher_meta = {}
            stats = {}

        elif rec == "info":
            if len(fields) >= 3:
                if fields[1] == "visteam":
                    vis_team = fields[2]
                elif fields[1] == "hometeam":
                    home_team = fields[2]
                elif fields[1] == "umphome":
                    ump_name = fields[2].strip('"')

        elif rec in ("start", "sub"):
            if len(fields) < 6:
                continue
            pid    = fields[1]
            pname  = fields[2].strip('"')
            side   = int(fields[3])   # 0=away, 1=home
            pos    = int(fields[5])   # 1 = pitcher

            if pos == 1:
                is_starter = (rec == "start")
                pitcher[side] = pid
                if (side, pid) not in pitcher_meta:
                    pitcher_meta[(side, pid)] = (pname, is_starter)
                if (side, pid) not in stats:
                    stats[(side, pid)] = [0, 0]

        elif rec == "play":
            if len(fields) < 7:
                continue
            batting_side  = int(fields[2])   # team at bat
            pitching_side = 1 - batting_side  # team pitching
            event_str = fields[6]

            if pitching_side not in pitcher:
                continue
            pid = pitcher[pitching_side]
            key = (pitching_side, pid)
            if key not in stats:
                stats[key] = [0, 0]

            stats[key][0] += count_outs(event_str)
            stats[key][1] += count_runs(event_str)

    flush_game()  # last game in file
    return records, ump_records


# ---------------------------------------------------------------------------
# Download + parse all years
# ---------------------------------------------------------------------------

def download_and_parse_year(year: int) -> tuple[list[dict], list[dict]]:
    """
    Download and parse one year of Retrosheet event files.
    Returns (pitcher_records, ump_records).
    """
    url = f"https://www.retrosheet.org/events/{year}eve.zip"
    print(f"  {year}...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        print(f"FAILED ({exc})")
        return [], []

    pitcher_records: list[dict] = []
    ump_records:     list[dict] = []

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        ev_files = [n for n in zf.namelist()
                    if n.endswith(".EVA") or n.endswith(".EVN")]
        for ev_name in ev_files:
            with zf.open(ev_name) as f:
                lines = [l.decode("latin-1") for l in f]
            p_recs, u_recs = parse_event_file(lines)
            pitcher_records.extend(p_recs)
            ump_records.extend(u_recs)

    print(f"{len(pitcher_records):,} pitcher records, {len(ump_records):,} ump records")
    return pitcher_records, ump_records


def fetch_all_event_data(years: list[int],
                         pitcher_cache: str,
                         ump_cache: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Incremental: only fetches years beyond the cached max.
    Saves both pitcher_game_logs.csv and umpire_game_logs.csv.
    Returns (pitcher_df, ump_df).
    """
    existing_p = existing_u = None
    fetch_years = list(years)

    if os.path.exists(pitcher_cache):
        existing_p = pd.read_csv(pitcher_cache, parse_dates=["game_date"])
        existing_p["_year"] = existing_p["game_date"].dt.year
        max_cached = int(existing_p["_year"].max())
        existing_p = existing_p.drop(columns=["_year"])
        fetch_years = [y for y in years if y > max_cached]
        if os.path.exists(ump_cache):
            existing_u = pd.read_csv(ump_cache, parse_dates=["game_date"])
        # Only skip fetching if BOTH caches exist and pitcher cache is current
        if not fetch_years and existing_u is not None:
            print(f"Event cache up to date (through {max_cached}); loading from cache")
            return existing_p, existing_u
        if not fetch_years and existing_u is None:
            # Pitcher cache is current but ump cache is missing — re-parse all years
            fetch_years = list(years)
            print(f"Pitcher cache up to date but ump cache missing; re-parsing {fetch_years}...")
        else:
            print(f"Event cache has through {max_cached}; fetching {fetch_years}...")

    all_pitcher: list[dict] = []
    all_ump:     list[dict] = []

    for year in fetch_years:
        p_recs, u_recs = download_and_parse_year(year)
        all_pitcher.extend(p_recs)
        all_ump.extend(u_recs)
        time.sleep(1.5)

    if not all_pitcher:
        return existing_p, existing_u

    new_p = pd.DataFrame(all_pitcher)
    new_u = pd.DataFrame(all_ump)

    # Only prepend existing pitcher data if we fetched INCREMENTAL years
    # (not a full re-parse triggered by missing ump cache — that would double data).
    is_incremental = existing_p is not None and set(fetch_years) != set(years)

    combined_p = pd.concat([existing_p, new_p], ignore_index=True) if is_incremental else new_p
    combined_u = pd.concat([existing_u, new_u], ignore_index=True) if existing_u is not None else new_u

    combined_p.to_csv(pitcher_cache, index=False)
    combined_u.to_csv(ump_cache,     index=False)
    print(f"\nSaved {len(combined_p):,} pitcher records to {pitcher_cache}")
    print(f"Saved {len(combined_u):,} umpire records to {ump_cache}")

    return combined_p, combined_u


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(pitcher_df: pd.DataFrame, ump_df: pd.DataFrame) -> None:
    print("\n=== Sanity Check ===")
    starters = pitcher_df[pitcher_df["is_starter"]]
    print(f"Total pitcher-game records : {len(pitcher_df):,}")
    print(f"Starter records            : {len(starters):,}")
    print(f"Reliever records           : {len(pitcher_df) - len(starters):,}")
    print(f"\nMedian starter outs        : {starters['outs_recorded'].median():.1f}  "
          f"(~{starters['outs_recorded'].median()/3:.1f} IP)")
    print(f"Median starter runs        : {starters['runs_allowed'].median():.1f}")
    print(f"\nSample starter records:")
    print(starters[["game_date","home_team","away_team",
                     "pitcher_name","team_side",
                     "outs_recorded","runs_allowed"]].head(5).to_string(index=False))
    if ump_df is not None:
        print(f"\nUmpire records             : {len(ump_df):,}")
        print(f"Unique umpires             : {ump_df['ump_name'].nunique()}")
        print(f"\nSample umpire records:")
        print(ump_df[["game_date","home_team","away_team","ump_name"]].head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    years        = list(range(2015, 2026))   # through 2025; Retrosheet won't have 2026 yet
    pitcher_cache = os.path.join(DATA_DIR, "pitcher_game_logs.csv")
    ump_cache     = os.path.join(DATA_DIR, "umpire_game_logs.csv")

    print("=== Downloading & parsing Retrosheet event files ===")
    pitcher_df, ump_df = fetch_all_event_data(years, pitcher_cache, ump_cache)
    sanity_check(pitcher_df, ump_df)
    print("\nDone. Run feature_engineering.py next.")
