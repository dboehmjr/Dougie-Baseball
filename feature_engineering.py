"""
Transform raw game logs + pitcher stats into a model-ready feature matrix.

Output: data/features.csv  (one row per game, home-team perspective)
Target: home_win  (1 = home team won, 0 = away team won)

Features used:
  - Rolling run-differential (15-game window, shift-1 to prevent leakage)
  - Rolling runs-scored per game (offensive proxy)
  - Season team ERA (weighted by GS, from Baseball Reference)
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
import pandas as pd
import numpy as np

# Import weather helpers (park CF bearings + wind projection)
from fetch_weather import PARK_CF_BEARING, FULL_DOME, wind_to_cf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Team name normalization
# Baseball Reference uses full city/nickname names in the Tm column.
# ---------------------------------------------------------------------------

BREF_NAME_TO_ABBREV = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL",
    "baltimore orioles": "BAL", "boston red sox": "BOS",
    "chicago cubs": "CHC", "chicago white sox": "CHW",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE",
    "cleveland indians": "CLE", "colorado rockies": "COL",
    "detroit tigers": "DET", "houston astros": "HOU",
    "kansas city royals": "KCR", "los angeles angels": "LAA",
    "los angeles dodgers": "LAD", "miami marlins": "MIA",
    "milwaukee brewers": "MIL", "minnesota twins": "MIN",
    "new york mets": "NYM", "new york yankees": "NYY",
    "oakland athletics": "OAK", "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT", "san diego padres": "SDP",
    "seattle mariners": "SEA", "san francisco giants": "SFG",
    "st. louis cardinals": "STL", "tampa bay rays": "TBR",
    "texas rangers": "TEX", "toronto blue jays": "TOR",
    "washington nationals": "WSN",
    # Short forms also in BRef
    "arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL",
    "boston": "BOS", "chicago": "CHC", "cincinnati": "CIN",
    "cleveland": "CLE", "colorado": "COL", "detroit": "DET",
    "houston": "HOU", "kansas city": "KCR", "la angels": "LAA",
    "la dodgers": "LAD", "miami": "MIA", "milwaukee": "MIL",
    "minnesota": "MIN", "ny mets": "NYM", "ny yankees": "NYY",
    "oakland": "OAK", "philadelphia": "PHI", "pittsburgh": "PIT",
    "san diego": "SDP", "seattle": "SEA", "san francisco": "SFG",
    "st. louis": "STL", "tampa bay": "TBR", "texas": "TEX",
    "toronto": "TOR", "washington": "WSN",
}

SCHEDULE_ABBREV_MAP = {
    "CWS": "CHW", "KC": "KCR", "SD": "SDP", "SF": "SFG",
    "TB": "TBR", "WSH": "WSN",
}


def normalize_schedule_team(name: str) -> str:
    s = str(name).strip().upper()
    return SCHEDULE_ABBREV_MAP.get(s, s)


def normalize_bref_team(name: str) -> str:
    s = str(name).strip().lower()
    return BREF_NAME_TO_ABBREV.get(s, s.upper())


# ---------------------------------------------------------------------------
# Ballpark coordinates (latitude, longitude) — used for travel distance
# ---------------------------------------------------------------------------

PARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4453, -112.0667),   # Chase Field, Phoenix AZ
    "ATL": (33.8908,  -84.4677),   # Truist Park, Cumberland GA
    "BAL": (39.2838,  -76.6218),   # Camden Yards, Baltimore MD
    "BOS": (42.3467,  -71.0972),   # Fenway Park, Boston MA
    "CHC": (41.9484,  -87.6553),   # Wrigley Field, Chicago IL
    "CHW": (41.8300,  -87.6338),   # Guaranteed Rate Field, Chicago IL
    "CIN": (39.0979,  -84.5082),   # Great American Ball Park, Cincinnati OH
    "CLE": (41.4962,  -81.6852),   # Progressive Field, Cleveland OH
    "COL": (39.7559, -104.9942),   # Coors Field, Denver CO
    "DET": (42.3390,  -83.0485),   # Comerica Park, Detroit MI
    "HOU": (29.7573,  -95.3555),   # Minute Maid Park, Houston TX
    "KCR": (39.0517,  -94.4803),   # Kauffman Stadium, Kansas City MO
    "LAA": (33.8003, -117.8827),   # Angel Stadium, Anaheim CA
    "LAD": (34.0739, -118.2400),   # Dodger Stadium, Los Angeles CA
    "MIA": (25.7781,  -80.2197),   # LoanDepot Park, Miami FL
    "MIL": (43.0280,  -87.9712),   # American Family Field, Milwaukee WI
    "MIN": (44.9817,  -93.2781),   # Target Field, Minneapolis MN
    "NYM": (40.7571,  -73.8458),   # Citi Field, Queens NY
    "NYY": (40.8296,  -73.9262),   # Yankee Stadium, Bronx NY
    "OAK": (37.7516, -122.2005),   # Oakland Coliseum, Oakland CA
    "PHI": (39.9061,  -75.1665),   # Citizens Bank Park, Philadelphia PA
    "PIT": (40.4469,  -80.0057),   # PNC Park, Pittsburgh PA
    "SDP": (32.7076, -117.1570),   # Petco Park, San Diego CA
    "SEA": (47.5914, -122.3325),   # T-Mobile Park, Seattle WA
    "SFG": (37.7786, -122.3893),   # Oracle Park, San Francisco CA
    "STL": (38.6226,  -90.1928),   # Busch Stadium, St. Louis MO
    "TBR": (27.7683,  -82.6534),   # Tropicana Field, St. Petersburg FL
    "TEX": (32.7473,  -97.0831),   # Globe Life Field, Arlington TX
    "TOR": (43.6414,  -79.3894),   # Rogers Centre, Toronto ON
    "WSN": (38.8730,  -77.0074),   # Nationals Park, Washington DC
}


# ---------------------------------------------------------------------------
# Park dimensions — physical park characteristics that affect scoring
# lf_dist / cf_dist / rf_dist: foul-pole/CF distances in feet
# lf_wall_ht: left-field wall height (feet) — affects HR rate
# altitude_ft: stadium elevation (feet) — thin air boosts batted-ball carry
# ---------------------------------------------------------------------------

PARK_DIMENSIONS: dict[str, dict] = {
    "ARI": {"lf_dist": 330, "cf_dist": 407, "rf_dist": 334, "lf_wall_ht":  7.5, "altitude_ft": 1082},
    "ATL": {"lf_dist": 335, "cf_dist": 400, "rf_dist": 325, "lf_wall_ht":  8.0, "altitude_ft": 1050},
    "BAL": {"lf_dist": 333, "cf_dist": 400, "rf_dist": 318, "lf_wall_ht":  7.0, "altitude_ft":   20},
    "BOS": {"lf_dist": 310, "cf_dist": 420, "rf_dist": 302, "lf_wall_ht": 37.2, "altitude_ft":   21},  # Green Monster
    "CHC": {"lf_dist": 355, "cf_dist": 400, "rf_dist": 353, "lf_wall_ht": 11.6, "altitude_ft":  595},
    "CHW": {"lf_dist": 330, "cf_dist": 400, "rf_dist": 335, "lf_wall_ht":  8.0, "altitude_ft":  595},
    "CIN": {"lf_dist": 328, "cf_dist": 404, "rf_dist": 325, "lf_wall_ht":  8.0, "altitude_ft":  550},
    "CLE": {"lf_dist": 325, "cf_dist": 405, "rf_dist": 325, "lf_wall_ht": 19.0, "altitude_ft":  655},
    "COL": {"lf_dist": 347, "cf_dist": 415, "rf_dist": 350, "lf_wall_ht":  8.0, "altitude_ft": 5200},  # Coors
    "DET": {"lf_dist": 345, "cf_dist": 420, "rf_dist": 330, "lf_wall_ht":  8.0, "altitude_ft":  600},
    "HOU": {"lf_dist": 315, "cf_dist": 409, "rf_dist": 326, "lf_wall_ht":  7.0, "altitude_ft":   43},
    "KCR": {"lf_dist": 330, "cf_dist": 410, "rf_dist": 330, "lf_wall_ht":  8.0, "altitude_ft":  750},
    "LAA": {"lf_dist": 330, "cf_dist": 400, "rf_dist": 330, "lf_wall_ht":  8.0, "altitude_ft":  160},
    "LAD": {"lf_dist": 330, "cf_dist": 395, "rf_dist": 330, "lf_wall_ht":  9.0, "altitude_ft":  515},
    "MIA": {"lf_dist": 344, "cf_dist": 416, "rf_dist": 335, "lf_wall_ht":  8.0, "altitude_ft":    6},
    "MIL": {"lf_dist": 344, "cf_dist": 400, "rf_dist": 345, "lf_wall_ht":  8.0, "altitude_ft":  635},
    "MIN": {"lf_dist": 339, "cf_dist": 411, "rf_dist": 328, "lf_wall_ht":  8.0, "altitude_ft":  841},
    "NYM": {"lf_dist": 335, "cf_dist": 408, "rf_dist": 330, "lf_wall_ht":  8.0, "altitude_ft":   20},
    "NYY": {"lf_dist": 318, "cf_dist": 408, "rf_dist": 314, "lf_wall_ht":  8.0, "altitude_ft":   55},
    "OAK": {"lf_dist": 330, "cf_dist": 400, "rf_dist": 330, "lf_wall_ht":  8.0, "altitude_ft":   25},
    "PHI": {"lf_dist": 329, "cf_dist": 401, "rf_dist": 330, "lf_wall_ht":  6.0, "altitude_ft":   20},
    "PIT": {"lf_dist": 325, "cf_dist": 399, "rf_dist": 320, "lf_wall_ht":  6.0, "altitude_ft":  730},
    "SDP": {"lf_dist": 336, "cf_dist": 396, "rf_dist": 322, "lf_wall_ht":  8.0, "altitude_ft":   20},
    "SEA": {"lf_dist": 331, "cf_dist": 401, "rf_dist": 326, "lf_wall_ht":  8.0, "altitude_ft":   17},
    "SFG": {"lf_dist": 339, "cf_dist": 399, "rf_dist": 309, "lf_wall_ht":  8.0, "altitude_ft":   10},
    "STL": {"lf_dist": 336, "cf_dist": 400, "rf_dist": 335, "lf_wall_ht":  8.0, "altitude_ft":  465},
    "TBR": {"lf_dist": 315, "cf_dist": 404, "rf_dist": 322, "lf_wall_ht":  8.0, "altitude_ft":   28},
    "TEX": {"lf_dist": 329, "cf_dist": 407, "rf_dist": 326, "lf_wall_ht":  8.0, "altitude_ft":  551},
    "TOR": {"lf_dist": 328, "cf_dist": 400, "rf_dist": 328, "lf_wall_ht":  8.0, "altitude_ft":  287},
    "WSN": {"lf_dist": 336, "cf_dist": 402, "rf_dist": 335, "lf_wall_ht":  8.0, "altitude_ft":   25},
}


def attach_park_dimensions(games: pd.DataFrame) -> pd.DataFrame:
    """Attach static park dimension columns to each game (keyed on home_team)."""
    for col in ["lf_dist", "cf_dist", "rf_dist", "lf_wall_ht", "altitude_ft"]:
        games[f"park_{col}"] = games["home_team"].map(
            {team: dims[col] for team, dims in PARK_DIMENSIONS.items()}
        )
    cov = games["park_cf_dist"].notna().mean()
    print(f"  Park dimensions coverage: {cov:.1%}")
    return games


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Step 1 – clean game logs
# ---------------------------------------------------------------------------

def clean_game_logs(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df = df[df["Home_Away"] == "Home"].copy()
    df = df[df["W/L"].str.contains("W|L", na=False)].copy()
    df = df[~df["W/L"].str.contains("susp|ppd", case=False, na=False)].copy()

    df["team"] = df["team"].apply(normalize_schedule_team)
    df["Opp"]  = df["Opp"].apply(normalize_schedule_team)

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str).str.extract(r"(\w+ \d+)")[0] + " " + df["year"].astype(str),
        format="%b %d %Y", errors="coerce"
    )
    df = df.dropna(subset=["Date"])
    df["home_win"] = df["W/L"].str.startswith("W").astype(int)

    return df[["Date", "year", "team", "Opp", "home_win", "R", "RA"]].rename(columns={
        "team": "home_team",
        "Opp":  "away_team",
        "R":    "home_runs",
        "RA":   "away_runs",
    })


# ---------------------------------------------------------------------------
# Step 2 – rolling run-differential and runs-scored per game (15-game window)
# ---------------------------------------------------------------------------

def _win_streak(run_diff_series: pd.Series) -> pd.Series:
    """
    Compute rolling win/loss streak after each game.
    Positive = win streak length, negative = losing streak length.
    Shift by 1 after calling to get the streak *going into* each game.
    """
    cur = 0
    streaks = []
    for val in run_diff_series:
        if pd.isna(val):
            streaks.append(0)
        elif val > 0:
            cur = max(1, cur + 1)
            streaks.append(cur)
        else:
            cur = min(-1, cur - 1)
            streaks.append(cur)
    return pd.Series(streaks, index=run_diff_series.index)


def add_rolling_stats(games: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    home = games[["Date", "home_team", "home_runs", "away_runs"]].copy()
    home.columns = ["Date", "team", "runs_for", "runs_against"]

    away = games[["Date", "away_team", "away_runs", "home_runs"]].copy()
    away.columns = ["Date", "team", "runs_for", "runs_against"]

    tg = pd.concat([home, away]).sort_values(["team", "Date"]).reset_index(drop=True)
    tg["run_diff"] = tg["runs_for"] - tg["runs_against"]

    grp = tg.groupby("team")

    # 15-game rolling window (existing)
    tg["rolling_run_diff"] = grp["run_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=5).mean()
    )
    tg["rolling_runs_scored"] = grp["runs_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=5).mean()
    )

    # 7-game rolling window — short-term hot/cold signal
    tg["last7_run_diff"] = grp["run_diff"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )

    # Win/loss streak going into each game (shift-1 of cumulative streak)
    tg["_streak_after"] = grp["run_diff"].transform(_win_streak)
    tg["streak"] = grp["_streak_after"].transform(lambda s: s.shift(1).fillna(0))
    tg = tg.drop(columns=["_streak_after"])

    def merge_side(side_col, rd_col, rs_col, rd7_col, streak_col):
        side = tg.merge(
            games[["Date", side_col]],
            left_on=["Date", "team"], right_on=["Date", side_col],
            how="inner"
        )[[side_col, "Date", "rolling_run_diff", "rolling_runs_scored",
           "last7_run_diff", "streak"]].rename(columns={
            "rolling_run_diff":    rd_col,
            "rolling_runs_scored": rs_col,
            "last7_run_diff":      rd7_col,
            "streak":              streak_col,
        }).drop_duplicates(["Date", side_col])
        return side

    home_stats = merge_side("home_team", "home_rolling_rd", "home_rolling_rs",
                             "home_last7_rd", "home_streak")
    away_stats = merge_side("away_team", "away_rolling_rd", "away_rolling_rs",
                             "away_last7_rd", "away_streak")

    games = games.merge(home_stats, on=["Date", "home_team"], how="left")
    games = games.merge(away_stats, on=["Date", "away_team"], how="left")
    return games


# ---------------------------------------------------------------------------
# Step 3 – per-game starting pitcher ERA
#
# We join Retrosheet SP assignments (game_sp.csv) against Baseball Reference
# pitcher season stats (pitcher_stats.csv) by name + team + year.
# Fall back to team ERA when a match isn't found.
# ---------------------------------------------------------------------------

# Common first-name nicknames: canonical (BRef) → short forms (Retrosheet)
_NICKNAME_MAP: dict[str, str] = {
    "matthew": "matt",
    "michael": "mike",
    "thomas":  "tom",
    "tommy":   "tom",
    "timothy": "tim",
    "william": "will",
    "robert":  "rob",
    "vincent": "vince",
    "stephen": "steve",
    "steven":  "steve",
    "nathaniel": "nate",
    "nicholas": "nick",
    "joshua":  "josh",
    "jonathan": "jon",
    "christopher": "chris",
    "alexander": "alex",
    "daniel":  "dan",
    "gregory": "greg",
    "benjamin": "ben",
    "zachary": "zach",
    "samuel":  "sam",
    "edward":  "ed",
    "joseph":  "joe",
    "andrew":  "andy",
    "anthony": "tony",
    "richard": "rich",
    "james":   "jim",
    "charles": "charlie",
    "ryan":    "ryan",  # no change, placeholder
}


def _strip_accents(name: str) -> str:
    """Remove diacritics (é→e, ó→o, ñ→n) and drop any remaining non-ASCII."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return ascii_only.encode("ascii", "ignore").decode("ascii")


def _fix_literal_escapes(name: str) -> str:
    """
    Handle names stored in CSV as literal escape text: 'Rodr\\xc3\\xadguez'
    → interpret each \\xNN pair as a byte, then decode the byte sequence as UTF-8.
    """
    if "\\x" not in name:
        return name
    try:
        # Replace each \xNN with the corresponding byte value (as a latin-1 char)
        with_bytes = re.sub(
            r"\\x([0-9a-fA-F]{2})",
            lambda m: chr(int(m.group(1), 16)),
            name,
        )
        # Those latin-1 chars are actually UTF-8 bytes — decode properly
        return with_bytes.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def _fix_mojibake(name: str) -> str:
    """Recover names where UTF-8 bytes were mis-read as latin-1 (e.g. JosÃ© → José)."""
    try:
        return name.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def _normalize_sp_name(name: str) -> str:
    """Standardize pitcher name for matching across Retrosheet and BRef."""
    name = str(name).strip()
    name = _fix_literal_escapes(name)  # handle BRef CSV literal \xNN sequences
    name = _fix_mojibake(name)         # handle mojibake (UTF-8 read as latin-1)
    name = _strip_accents(name)        # é→e, ó→o, any remaining non-ASCII dropped
    name = name.lower()
    # Retrosheet: "Last, First" → "first last"
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    # Drop suffixes
    for suffix in [" jr.", " sr.", " ii", " iii", " iv"]:
        name = name.replace(suffix, "")
    name = name.replace("-", " ")  # hyun-jin → hyun jin
    name = " ".join(name.split())  # collapse whitespace
    # Expand long first names to their common short form
    parts = name.split(" ", 1)
    if parts:
        parts[0] = _NICKNAME_MAP.get(parts[0], parts[0])
        name = " ".join(parts)
    return name


def _park_adjust_era(era: float, park_factor: float) -> float:
    """
    Remove the home-park bias from a pitcher's season ERA.

    A starter pitches ~50% of games at home, so their raw ERA reflects
    half a season in their home park. Adjustment formula:
        ERA_adj = ERA × (2 / (1 + PF))
    When PF=1.35 (Coors): ERA_adj ≈ ERA × 0.851  → pitcher looks better
    When PF=0.92 (Petco): ERA_adj ≈ ERA × 1.042  → pitcher looks slightly worse
    """
    if np.isnan(park_factor) or park_factor <= 0:
        return era
    return era * (2.0 / (1.0 + park_factor))


def build_pitcher_era_lookup(pitcher_stats: pd.DataFrame,
                             park_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame keyed by (name_norm, year) with both raw ERA
    and park-adjusted ERA (ERA_adj).
    """
    sp = pitcher_stats.copy()
    sp["team"]      = sp["Tm"].apply(normalize_bref_team)
    sp["GS"]        = pd.to_numeric(sp["GS"], errors="coerce").fillna(0)
    sp["ERA"]       = pd.to_numeric(sp["ERA"], errors="coerce")
    sp["name_norm"] = sp["Name"].apply(_normalize_sp_name)
    sp = sp[sp["GS"] >= 1].dropna(subset=["ERA"])
    sp = sp.sort_values("GS", ascending=False).drop_duplicates(["name_norm", "year"])

    # Join team park factor for the pitcher's home park
    sp = sp.merge(
        park_factors.rename(columns={"home_team": "team"}),
        on=["team", "year"], how="left"
    )
    sp["ERA_adj"] = sp.apply(
        lambda r: _park_adjust_era(r["ERA"], r.get("park_factor", np.nan)), axis=1
    )
    return sp[["name_norm", "team", "year", "ERA", "ERA_adj", "GS"]]


def build_team_era_fallback(pitcher_stats: pd.DataFrame,
                            park_factors: pd.DataFrame) -> pd.DataFrame:
    """Season-level team ERA (raw + adjusted) weighted by GS — fallback when SP name misses."""
    sp = pitcher_stats.copy()
    sp["team"] = sp["Tm"].apply(normalize_bref_team)
    sp["GS"]   = pd.to_numeric(sp["GS"], errors="coerce").fillna(0)
    sp["ERA"]  = pd.to_numeric(sp["ERA"], errors="coerce")
    sp = sp[sp["GS"] >= 1].dropna(subset=["ERA"])

    sp = sp.merge(
        park_factors.rename(columns={"home_team": "team"}),
        on=["team", "year"], how="left"
    )
    sp["ERA_adj"] = sp.apply(
        lambda r: _park_adjust_era(r["ERA"], r.get("park_factor", np.nan)), axis=1
    )

    def wavg(g, col):
        return np.average(g[col], weights=g["GS"].clip(lower=1))

    team_era = (
        sp.groupby(["team", "year"])
        .apply(lambda g: pd.Series({
            "team_era":     wavg(g, "ERA"),
            "team_era_adj": wavg(g, "ERA_adj"),
        }))
        .reset_index()
    )
    return team_era


def attach_sp_era(games: pd.DataFrame,
                  game_sp: pd.DataFrame,
                  pitcher_era: pd.DataFrame,
                  team_era_fallback: pd.DataFrame) -> pd.DataFrame:
    """
    Join Retrosheet SP assignments to each game, look up raw + park-adjusted ERA.
    Falls back to team ERA when SP name is not found.
    """
    gs = game_sp.copy()
    gs["home_sp_norm"] = gs["home_sp_name"].apply(_normalize_sp_name)
    gs["away_sp_norm"] = gs["away_sp_name"].apply(_normalize_sp_name)

    games = games.merge(
        gs[["Date", "home_team", "away_team", "home_sp_norm", "away_sp_norm"]],
        on=["Date", "home_team", "away_team"], how="left"
    )

    era_raw = pitcher_era.set_index(["name_norm", "year"])["ERA"]
    era_adj = pitcher_era.set_index(["name_norm", "year"])["ERA_adj"]

    def lookup(row, sp_col, lookup_series):
        name = row.get(sp_col)
        year = row.get("year")
        if pd.isna(name) or pd.isna(year):
            return np.nan
        return lookup_series.get((name, int(year)), np.nan)

    games["home_sp_era"]     = games.apply(lookup, sp_col="home_sp_norm", lookup_series=era_raw, axis=1)
    games["away_sp_era"]     = games.apply(lookup, sp_col="away_sp_norm", lookup_series=era_raw, axis=1)
    games["home_sp_era_adj"] = games.apply(lookup, sp_col="home_sp_norm", lookup_series=era_adj, axis=1)
    games["away_sp_era_adj"] = games.apply(lookup, sp_col="away_sp_norm", lookup_series=era_adj, axis=1)

    # Fallback to team ERA
    fb = team_era_fallback.copy()
    games = games.merge(
        fb.rename(columns={"team": "home_team", "team_era": "home_team_era",
                            "team_era_adj": "home_team_era_adj"}),
        on=["home_team", "year"], how="left"
    )
    games = games.merge(
        fb.rename(columns={"team": "away_team", "team_era": "away_team_era",
                            "team_era_adj": "away_team_era_adj"}),
        on=["away_team", "year"], how="left"
    )

    games["home_sp_era"]     = games["home_sp_era"].fillna(games["home_team_era"])
    games["away_sp_era"]     = games["away_sp_era"].fillna(games["away_team_era"])
    games["home_sp_era_adj"] = games["home_sp_era_adj"].fillna(games["home_team_era_adj"])
    games["away_sp_era_adj"] = games["away_sp_era_adj"].fillna(games["away_team_era_adj"])

    sp_hit_rate = games["home_sp_era_adj"].notna().mean()
    name_match  = (~games["home_sp_norm"].isna() &
                   games.apply(lambda r: not np.isnan(
                       era_adj.get((r["home_sp_norm"], int(r["year"]))
                                   if not pd.isna(r["home_sp_norm"]) else ("", 0),
                                   np.nan)), axis=1)).mean()
    print(f"  SP ERA coverage : {sp_hit_rate:.1%}")
    print(f"  SP name match   : {name_match:.1%}")

    return games


# ---------------------------------------------------------------------------
# Step 4 – park factors
#
# Park factor = (runs/game at home) / (runs/game away) for each team.
# A PF > 1.0 means the park inflates scoring (e.g. Coors Field ~1.20).
# A PF < 1.0 suppresses scoring (e.g. Petco Park ~0.92).
#
# We use a 3-year rolling window of PRIOR seasons to avoid data leakage.
# The home team's park factor is the only one that matters here since
# both teams play in the same ballpark.
# ---------------------------------------------------------------------------

def compute_park_factors(games: pd.DataFrame, window_years: int = 3) -> pd.DataFrame:
    """
    For each (team, year) compute PF from the prior `window_years` seasons.
    Returns a DataFrame with columns: home_team, year, park_factor.
    """
    # runs per game at home and away, per team per year
    home_rpg = (
        games.groupby(["home_team", "year"])
        .apply(lambda g: (g["home_runs"] + g["away_runs"]).mean())
        .reset_index(name="home_rpg")
    )

    # away view: each home game is also an "away" game for the opponent
    away_rpg = (
        games.groupby(["away_team", "year"])
        .apply(lambda g: (g["home_runs"] + g["away_runs"]).mean())
        .reset_index(name="away_rpg")
        .rename(columns={"away_team": "home_team"})
    )

    pf_annual = home_rpg.merge(away_rpg, on=["home_team", "year"], how="inner")
    pf_annual["annual_pf"] = pf_annual["home_rpg"] / pf_annual["away_rpg"]
    pf_annual = pf_annual.sort_values(["home_team", "year"])

    # Rolling mean over prior seasons (shift by 1 to exclude current year)
    pf_annual["park_factor"] = (
        pf_annual.groupby("home_team")["annual_pf"]
        .transform(lambda s: s.shift(1).rolling(window_years, min_periods=1).mean())
    )

    return pf_annual[["home_team", "year", "park_factor"]]


def attach_park_factor(games: pd.DataFrame) -> pd.DataFrame:
    pf = compute_park_factors(games)
    games = games.merge(pf, on=["home_team", "year"], how="left")
    coverage = games["park_factor"].notna().mean()
    print(f"  Park factor coverage: {coverage:.1%}")
    return games


# ---------------------------------------------------------------------------
# SP pitch stuff (FanGraphs) — fastball velocity, whiff rate, K%, xFIP
#
# Joined at the season level: for game in year Y, we use the SP's FanGraphs
# stats from year Y.  Falls back to team median.  Throws (handedness) is stored
# separately so predict.py can look up batting splits.
# ---------------------------------------------------------------------------

def attach_sp_stuff(games: pd.DataFrame,
                    pitcher_stuff: pd.DataFrame) -> pd.DataFrame:
    """
    Join FanGraphs SP stuff (FBv, SwStr%, K%, xFIP, Throws) to every game.
    Requires home_sp_norm / away_sp_norm columns (added by attach_sp_era).
    Falls back to team median when a name lookup misses.
    """
    if pitcher_stuff is None or pitcher_stuff.empty:
        for col in ["home_sp_fbv", "away_sp_fbv", "home_sp_swstr", "away_sp_swstr",
                    "home_sp_k_pct", "away_sp_k_pct", "home_sp_xfip", "away_sp_xfip",
                    "home_sp_throws", "away_sp_throws", "home_sp_pa", "away_sp_pa"]:
            games[col] = np.nan
        print("  SP stuff: pitcher_stuff.csv not found — all NaN")
        return games

    stuff = pitcher_stuff.copy()
    # Pre-compute team medians per year for fallback
    num_cols = ["FBv", "SwStr_pct", "K_pct", "xFIP"]
    team_med = (stuff.groupby(["team", "year"])[num_cols]
                .median().reset_index())

    # Build fast lookup dicts
    def _make_lookup(col):
        return stuff.dropna(subset=["name_norm", col]).set_index(
            ["name_norm", "year"])[col].to_dict()

    fbv_lk     = _make_lookup("FBv")
    swstr_lk   = _make_lookup("SwStr_pct")
    kpct_lk    = _make_lookup("K_pct")
    xfip_lk    = _make_lookup("xFIP")
    pa_lk      = _make_lookup("pa")
    throws_lk  = (stuff.dropna(subset=["name_norm", "Throws"])
                       .set_index(["name_norm", "year"])["Throws"].to_dict()
                  if "Throws" in stuff.columns else {})

    team_fbv    = team_med.set_index(["team", "year"])["FBv"].to_dict()
    team_swstr  = team_med.set_index(["team", "year"])["SwStr_pct"].to_dict()
    team_kpct   = team_med.set_index(["team", "year"])["K_pct"].to_dict()
    team_xfip   = team_med.set_index(["team", "year"])["xFIP"].to_dict()

    def _lookup(name_norm, team, year, direct_lk, fallback_lk):
        key = (name_norm, year)
        v = direct_lk.get(key, np.nan)
        if pd.isna(v):
            v = fallback_lk.get((team, year), np.nan)
        return v

    for side, sp_col, team_col in [("home", "home_sp_norm", "home_team"),
                                    ("away", "away_sp_norm", "away_team")]:
        yr   = games["year"]
        name = games[sp_col] if sp_col in games.columns else pd.Series([None]*len(games))
        team = games[team_col]

        games[f"{side}_sp_fbv"]    = [_lookup(n, t, y, fbv_lk,    team_fbv)
                                       for n, t, y in zip(name, team, yr)]
        games[f"{side}_sp_swstr"]  = [_lookup(n, t, y, swstr_lk,  team_swstr)
                                       for n, t, y in zip(name, team, yr)]
        games[f"{side}_sp_k_pct"]  = [_lookup(n, t, y, kpct_lk,   team_kpct)
                                       for n, t, y in zip(name, team, yr)]
        games[f"{side}_sp_xfip"]   = [_lookup(n, t, y, xfip_lk,   team_xfip)
                                       for n, t, y in zip(name, team, yr)]
        games[f"{side}_sp_pa"]     = [pa_lk.get((n, y), np.nan) if n else np.nan
                                       for n, y in zip(name, yr)]
        games[f"{side}_sp_throws"] = [throws_lk.get((n, y)) if n else None
                                       for n, y in zip(name, yr)]

    cov = games["home_sp_fbv"].notna().mean()
    print(f"  SP stuff coverage: {cov:.1%}  (FBv)")
    return games


# ---------------------------------------------------------------------------
# Step 5 – rest days and travel distance
#
# For each game we compute, using only the team's PRIOR game:
#   days_rest    = (current_date – prev_game_date).days – 1
#                  (0 = back-to-back, 1 = one off day, capped at 7)
#   travel_miles = great-circle distance from previous game's park
#                  to current game's park  (0 if already in same city)
#
# New features added to the model:
#   home_days_rest, away_days_rest, rest_diff
#   away_travel_miles, travel_diff  (away – home miles)
# ---------------------------------------------------------------------------

def compute_rest_and_travel(games: pd.DataFrame) -> pd.DataFrame:
    """
    Attach days-of-rest and travel-distance features to every game.
    Both teams' previous game location is always the home team's park
    of that prior game.
    """
    # Build a flat schedule: one row per (team, game)
    home_rows = pd.DataFrame({
        "Date":     games["Date"],
        "team":     games["home_team"],
        "location": games["home_team"],   # playing at their own park
    })
    away_rows = pd.DataFrame({
        "Date":     games["Date"],
        "team":     games["away_team"],
        "location": games["home_team"],   # at the home team's park
    })

    schedule = (
        pd.concat([home_rows, away_rows], ignore_index=True)
        .sort_values(["team", "Date"])
        # For doubleheaders keep only one row per (team, date) — good enough
        .drop_duplicates(subset=["team", "Date"], keep="first")
        .reset_index(drop=True)
    )

    grp = schedule.groupby("team")
    schedule["prev_date"]     = grp["Date"].transform(lambda s: s.shift(1))
    schedule["prev_location"] = grp["location"].transform(lambda s: s.shift(1))

    # Days rest — clip to [0, 7]; NaN for a team's very first game
    schedule["days_rest"] = (
        (schedule["Date"] - schedule["prev_date"]).dt.days - 1
    ).clip(lower=0, upper=7)
    schedule.loc[schedule["prev_date"].isna(), "days_rest"] = np.nan

    # Travel distance
    def _travel(row) -> float:
        prev = row["prev_location"]
        curr = row["location"]
        if prev != prev or prev == curr:   # NaN check or same park
            return 0.0
        c1 = PARK_COORDS.get(prev)
        c2 = PARK_COORDS.get(curr)
        if c1 is None or c2 is None:
            return np.nan
        return haversine_miles(c1[0], c1[1], c2[0], c2[1])

    schedule["travel_miles"] = schedule.apply(_travel, axis=1)

    # Merge home stats
    home_sched = (
        schedule
        .rename(columns={
            "team":         "home_team",
            "days_rest":    "home_days_rest",
            "travel_miles": "home_travel_miles",
        })
        [["Date", "home_team", "home_days_rest", "home_travel_miles"]]
    )
    # Merge away stats
    away_sched = (
        schedule
        .rename(columns={
            "team":         "away_team",
            "days_rest":    "away_days_rest",
            "travel_miles": "away_travel_miles",
        })
        [["Date", "away_team", "away_days_rest", "away_travel_miles"]]
    )

    games = games.merge(home_sched, on=["Date", "home_team"], how="left")
    games = games.merge(away_sched, on=["Date", "away_team"], how="left")

    cov = games["home_days_rest"].notna().mean()
    print(f"  Rest/travel coverage: {cov:.1%}")
    return games


# ---------------------------------------------------------------------------
# Step 7 – in-season rolling pitcher stats from Retrosheet event files
#
# For each game we compute, using only data from PRIOR games that season:
#   SP:      rolling ERA over last 5 starts  (outs→IP, runs allowed)
#   Bullpen: rolling RA/9 over last 15 days  (team relievers combined)
#
# Park-adjustment uses the same formula as season-level ERA above.
# ---------------------------------------------------------------------------

def _ra9(runs: float, outs: float) -> float:
    """Runs-allowed per 9 innings from raw outs and runs."""
    if outs < 1:
        return np.nan
    return (runs / outs) * 27.0   # 27 outs = 9 innings


def compute_inseason_sp_era(game_logs: pd.DataFrame,
                             game_sp: pd.DataFrame,
                             park_factors: pd.DataFrame,
                             n_starts: int = 5) -> pd.DataFrame:
    """
    For each game, compute the starting pitcher's rolling RA/9 over
    their last `n_starts` starts (shift-1 so current game is excluded).

    Returns: DataFrame with Date, home_team, away_team,
             home_sp_inseason_era, away_sp_inseason_era
    """
    starters = game_logs[game_logs["is_starter"]].copy()
    starters["game_date"] = pd.to_datetime(starters["game_date"])
    starters = starters.sort_values(["pitcher_id", "game_date"])

    # Rolling sum over last n_starts (shift 1 to exclude current game)
    grp = starters.groupby("pitcher_id")
    starters["roll_runs"] = grp["runs_allowed"].transform(
        lambda s: s.shift(1).rolling(n_starts, min_periods=2).sum()
    )
    starters["roll_outs"] = grp["outs_recorded"].transform(
        lambda s: s.shift(1).rolling(n_starts, min_periods=2).sum()
    )
    starters["sp_inseason_ra9"] = starters.apply(
        lambda r: _ra9(r["roll_runs"], r["roll_outs"]), axis=1
    )

    # Park-adjust: RA9 → park-adjusted RA9
    pf_lookup = park_factors.set_index(["home_team", "year"])["park_factor"]

    def park_adj_sp(row):
        ra9 = row["sp_inseason_ra9"]
        if np.isnan(ra9):
            return np.nan
        # Pitcher's home park — use their team_side + game teams
        pitcher_team = row["home_team"] if row["team_side"] == 1 else row["away_team"]
        year = row["game_date"].year
        pf = pf_lookup.get((pitcher_team, year), np.nan)
        return _park_adjust_era(ra9, pf)

    starters["sp_inseason_era_adj"] = starters.apply(park_adj_sp, axis=1)

    # Merge SP assignments to get pitcher_id per game
    gs = game_sp.copy()
    gs["Date"] = pd.to_datetime(gs["Date"])

    # Join home SP stats
    home_sp = starters[["game_date", "pitcher_id", "sp_inseason_era_adj"]].rename(
        columns={"game_date": "Date", "sp_inseason_era_adj": "home_sp_inseason_era"}
    )
    result = gs.merge(
        home_sp,
        left_on=["Date", "home_sp_id"],
        right_on=["Date", "pitcher_id"],
        how="left"
    ).drop(columns=["pitcher_id"])

    # Join away SP stats
    away_sp = starters[["game_date", "pitcher_id", "sp_inseason_era_adj"]].rename(
        columns={"game_date": "Date", "sp_inseason_era_adj": "away_sp_inseason_era"}
    )
    result = result.merge(
        away_sp,
        left_on=["Date", "away_sp_id"],
        right_on=["Date", "pitcher_id"],
        how="left"
    ).drop(columns=["pitcher_id"])

    coverage = result["home_sp_inseason_era"].notna().mean()
    print(f"  SP in-season ERA coverage  : {coverage:.1%}")
    return result[["Date", "home_team", "away_team",
                   "home_sp_inseason_era", "away_sp_inseason_era"]]


def compute_inseason_bullpen_era(game_logs: pd.DataFrame,
                                  park_factors: pd.DataFrame,
                                  window_days: int = 15) -> pd.DataFrame:
    """
    For each (team, game_date), compute the bullpen's rolling RA/9 over
    the prior `window_days` days using only relief appearances.
    shift-1 on game_date so current game is excluded.

    Returns: DataFrame with game_date, team, bullpen_inseason_era
    """
    relievers = game_logs[~game_logs["is_starter"]].copy()
    relievers["game_date"] = pd.to_datetime(relievers["game_date"])

    # Assign team per appearance
    relievers["team"] = np.where(
        relievers["team_side"] == 1,
        relievers["home_team"],
        relievers["away_team"]
    )

    # Sum runs and outs per team per game (all relievers combined)
    daily = (
        relievers.groupby(["team", "game_date"])
        .agg(runs=("runs_allowed", "sum"), outs=("outs_recorded", "sum"))
        .reset_index()
        .sort_values(["team", "game_date"])
    )

    # Rolling window over prior days — shift by 1 game to exclude current
    grp = daily.groupby("team")
    daily["roll_runs"] = grp["runs"].transform(
        lambda s: s.shift(1).rolling(window_days, min_periods=5).sum()
    )
    daily["roll_outs"] = grp["outs"].transform(
        lambda s: s.shift(1).rolling(window_days, min_periods=5).sum()
    )
    daily["bullpen_inseason_ra9"] = daily.apply(
        lambda r: _ra9(r["roll_runs"], r["roll_outs"]), axis=1
    )

    # Park-adjust
    pf_lookup = park_factors.set_index(["home_team", "year"])["park_factor"]

    def park_adj_bp(row):
        ra9 = row["bullpen_inseason_ra9"]
        if np.isnan(ra9):
            return np.nan
        year = row["game_date"].year
        pf = pf_lookup.get((row["team"], year), np.nan)
        return _park_adjust_era(ra9, pf)

    daily["bullpen_inseason_era"] = daily.apply(park_adj_bp, axis=1)

    coverage = daily["bullpen_inseason_era"].notna().mean()
    print(f"  Bullpen in-season ERA coverage: {coverage:.1%}")
    return daily[["team", "game_date", "bullpen_inseason_era"]]


def compute_bullpen_usage(game_logs: pd.DataFrame,
                          window_days: int = 3) -> pd.DataFrame:
    """
    For each (team, game_date), sum outs recorded by relievers in the prior
    `window_days` CALENDAR days (not game days).  Uses closed="left" rolling
    so the current game is always excluded.

    A higher number means the bullpen has been worked harder recently and
    may be fatigued / short on available arms.

    Returns: DataFrame with team, game_date, bullpen_outs_Xd
    """
    relievers = game_logs[~game_logs["is_starter"]].copy()
    relievers["game_date"] = pd.to_datetime(relievers["game_date"])
    relievers["team"] = np.where(
        relievers["team_side"] == 1,
        relievers["home_team"],
        relievers["away_team"],
    )

    # Sum outs per (team, date) across all relievers
    daily = (
        relievers.groupby(["team", "game_date"])
        .agg(outs=("outs_recorded", "sum"))
        .reset_index()
        .sort_values(["team", "game_date"])
    )

    col = f"bullpen_outs_{window_days}d"
    parts = []
    for _, grp in daily.groupby("team"):
        grp = grp.set_index("game_date").sort_index()
        # closed="left": window is [date - Xdays, date) — excludes current game
        grp[col] = grp["outs"].rolling(f"{window_days}D", closed="left").sum()
        parts.append(grp.reset_index())

    result = pd.concat(parts, ignore_index=True)
    cov = result[col].notna().mean()
    print(f"  Bullpen usage ({window_days}d) coverage: {cov:.1%}")
    return result[["team", "game_date", col]]


def attach_inseason_stats(games: pd.DataFrame,
                          game_logs: pd.DataFrame,
                          game_sp: pd.DataFrame,
                          park_factors: pd.DataFrame) -> pd.DataFrame:
    """Attach in-season rolling SP ERA, bullpen ERA, and bullpen usage to games."""

    # --- SP ERA ---
    sp_stats = compute_inseason_sp_era(game_logs, game_sp, park_factors)
    games = games.merge(sp_stats, on=["Date", "home_team", "away_team"], how="left")

    # --- Bullpen ERA (quality) ---
    bp_era = compute_inseason_bullpen_era(game_logs, park_factors)
    games = games.merge(
        bp_era.rename(columns={"team": "home_team", "game_date": "Date",
                                "bullpen_inseason_era": "home_bullpen_inseason_era"}),
        on=["Date", "home_team"], how="left"
    )
    games = games.merge(
        bp_era.rename(columns={"team": "away_team", "game_date": "Date",
                                "bullpen_inseason_era": "away_bullpen_inseason_era"}),
        on=["Date", "away_team"], how="left"
    )

    # --- Bullpen usage / fatigue (last 3 calendar days) ---
    bp_use = compute_bullpen_usage(game_logs, window_days=3)
    games = games.merge(
        bp_use.rename(columns={"team": "home_team", "game_date": "Date",
                                "bullpen_outs_3d": "home_bullpen_outs_3d"}),
        on=["Date", "home_team"], how="left"
    )
    games = games.merge(
        bp_use.rename(columns={"team": "away_team", "game_date": "Date",
                                "bullpen_outs_3d": "away_bullpen_outs_3d"}),
        on=["Date", "away_team"], how="left"
    )

    return games


# ---------------------------------------------------------------------------
# Step 8 – team batting quality (prior-season OPS)
#
# OPS (on-base + slugging) captures lineup quality independently of the
# rolling run-differential already in the model.  We use the PRIOR season's
# team OPS (PA-weighted average) so there is no data leakage.
#
# Source: Baseball Reference individual batting stats, aggregated per team.
# Multi-team (traded player) combined rows are excluded; each player's stats
# are attributed to the single team they played for.
# ---------------------------------------------------------------------------

def _normalize_batting_team(tm: str, lev: str) -> str:
    """
    Map BRef batting-stats team name → our 3-letter abbreviation.
    Disambiguates cities with two teams using the league in the Lev column.
    """
    tm_l  = str(tm).strip().lower()
    lev_l = str(lev).strip().lower()

    # Ambiguous city names: use league to distinguish
    if tm_l == "chicago":
        tm_l = "chicago white sox" if "al" in lev_l else "chicago cubs"
    elif tm_l == "new york":
        tm_l = "new york yankees" if "al" in lev_l else "new york mets"
    elif tm_l == "los angeles":
        tm_l = "los angeles angels" if "al" in lev_l else "los angeles dodgers"

    return BREF_NAME_TO_ABBREV.get(tm_l, tm_l.upper())


def compute_team_batting_quality(batting_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual batter rows into one PA-weighted OPS per (team, year).

    Filters:
      - MLB rows only  (Lev starts with "Maj")
      - Single-team rows only  (no comma in Tm — excludes traded-player totals)
      - Minimum 10 PA
    """
    df = batting_stats.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df[
        df["Lev"].str.startswith("Maj", na=False)
        & ~df["Tm"].str.contains(",", na=False)
    ].copy()

    df["PA"]  = pd.to_numeric(df["PA"],  errors="coerce").fillna(0)
    df["OPS"] = pd.to_numeric(df["OPS"], errors="coerce")
    df = df[(df["PA"] >= 10)].dropna(subset=["OPS", "year"])

    df["team"] = df.apply(
        lambda r: _normalize_batting_team(r["Tm"], r["Lev"]), axis=1
    )

    team_ops = (
        df.groupby(["team", "year"])
        .apply(lambda g: pd.Series({
            "team_ops": float(np.average(g["OPS"], weights=g["PA"].clip(lower=1)))
        }))
        .reset_index()
    )
    return team_ops


def attach_statcast_batting(games: pd.DataFrame,
                            statcast_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Attach prior-season Statcast batting metrics (barrel rate, hard hit%) to games.
    For a game in year Y, uses metrics from year Y-1 (no leakage — same pattern
    as team OPS).  Falls back to NaN when data is unavailable.
    """
    if statcast_df is None or statcast_df.empty:
        for col in ["home_barrel_pct", "away_barrel_pct",
                    "home_hard_hit_pct", "away_hard_hit_pct"]:
            games[col] = np.nan
        print("  Statcast batting: not found — all NaN")
        return games

    # Shift: stats from year Y go on games in year Y+1
    shifted = statcast_df.copy()
    shifted["year"] = shifted["year"] + 1

    bbl  = shifted.set_index(["team", "year"])["barrel_pct"].to_dict()
    hh   = shifted.set_index(["team", "year"])["hard_hit_pct"].to_dict() \
           if "hard_hit_pct" in shifted.columns else {}

    games["home_barrel_pct"]   = [bbl.get((t, y), np.nan)
                                   for t, y in zip(games["home_team"], games["year"])]
    games["away_barrel_pct"]   = [bbl.get((t, y), np.nan)
                                   for t, y in zip(games["away_team"], games["year"])]
    games["home_hard_hit_pct"] = [hh.get((t, y), np.nan)
                                   for t, y in zip(games["home_team"], games["year"])]
    games["away_hard_hit_pct"] = [hh.get((t, y), np.nan)
                                   for t, y in zip(games["away_team"], games["year"])]

    cov = games["home_barrel_pct"].notna().mean()
    print(f"  Statcast batting coverage: {cov:.1%}  (barrel rate)")
    return games


def attach_batting_quality(games: pd.DataFrame,
                           batting_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Attach PRIOR-year team OPS to each game.
    For a game in year Y, use team OPS from year Y-1 (no leakage).
    """
    team_ops = compute_team_batting_quality(batting_stats)

    # Shift: team OPS from year Y goes on games in year Y+1
    shifted = team_ops.copy()
    shifted["year"] = shifted["year"] + 1

    games = games.merge(
        shifted.rename(columns={"team": "home_team", "team_ops": "home_team_ops"}),
        on=["home_team", "year"], how="left",
    )
    games = games.merge(
        shifted.rename(columns={"team": "away_team", "team_ops": "away_team_ops"}),
        on=["away_team", "year"], how="left",
    )

    cov = games["home_team_ops"].notna().mean()
    print(f"  Batting quality coverage: {cov:.1%}")
    return games


# ---------------------------------------------------------------------------
# Step 9b – prior-season Pythagorean win%
#
# Pythagorean win% = RS² / (RS² + RA²) per team per season.
# We use year-1 so there's no leakage: a 2026 game gets 2025 win%.
# This anchors team quality for teams on unsustainable hot/cold streaks.
# ---------------------------------------------------------------------------

def attach_prior_win_pct(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pythagorean win% (RS²/(RS²+RA²)) per team per season from the
    games DataFrame itself, then join year-1 values as home/away features.
    """
    # Build a long-form view: one row per team per game with RS and RA
    home_view = games[["Date", "year", "home_team", "home_runs", "away_runs"]].copy()
    home_view.columns = ["Date", "year", "team", "rs", "ra"]
    away_view = games[["Date", "year", "away_team", "away_runs", "home_runs"]].copy()
    away_view.columns = ["Date", "year", "team", "rs", "ra"]
    long = pd.concat([home_view, away_view], ignore_index=True)

    season = (long.groupby(["team", "year"])
                  .agg(rs=("rs", "sum"), ra=("ra", "sum"))
                  .reset_index())
    season["pyth_wp"] = season["rs"] ** 2 / (season["rs"] ** 2 + season["ra"] ** 2)

    # Shift: season Y win% → games in year Y+1
    shifted = season[["team", "year", "pyth_wp"]].copy()
    shifted["year"] = shifted["year"] + 1

    games = games.merge(
        shifted.rename(columns={"team": "home_team", "pyth_wp": "home_prior_win_pct"}),
        on=["home_team", "year"], how="left",
    )
    games = games.merge(
        shifted.rename(columns={"team": "away_team", "pyth_wp": "away_prior_win_pct"}),
        on=["away_team", "year"], how="left",
    )

    cov = games["home_prior_win_pct"].notna().mean()
    print(f"  Prior-season Pythagorean win% coverage: {cov:.1%}")
    return games


# ---------------------------------------------------------------------------
# Step 10 – head-to-head matchup history
#
# For each game (home_team H vs away_team A) we look back at the last
# N meetings between H and A — regardless of who was home — and compute:
#   h2h_home_win_rate  — fraction H won across those N games (0–1)
#   h2h_run_diff       — avg run diff from H's perspective in those N games
#
# shift(1) ensures the current game is always excluded (no leakage).
# min_periods=1 so early-season games still get a value once one prior
# meeting exists; NaN only for the very first time two teams ever meet.
# ---------------------------------------------------------------------------

def compute_h2h_stats(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Attach rolling head-to-head win rate and run differential to every game.
    """
    games = games.sort_values("Date").reset_index(drop=True)

    # Build a symmetric view: one row per (focal_team, opponent, game)
    home_view = games[["Date", "home_team", "away_team", "home_win",
                        "home_runs", "away_runs"]].copy()
    home_view["focal"]     = home_view["home_team"]
    home_view["opponent"]  = home_view["away_team"]
    home_view["focal_win"] = home_view["home_win"]
    home_view["focal_rd"]  = home_view["home_runs"] - home_view["away_runs"]

    away_view = games[["Date", "home_team", "away_team", "home_win",
                        "home_runs", "away_runs"]].copy()
    away_view["focal"]     = away_view["away_team"]
    away_view["opponent"]  = away_view["home_team"]
    away_view["focal_win"] = 1 - away_view["home_win"]
    away_view["focal_rd"]  = away_view["away_runs"] - away_view["home_runs"]

    hist = (
        pd.concat(
            [home_view[["Date", "focal", "opponent", "focal_win", "focal_rd"]],
             away_view[["Date", "focal", "opponent", "focal_win", "focal_rd"]]],
            ignore_index=True,
        )
        .sort_values(["focal", "opponent", "Date"])
        .reset_index(drop=True)
    )

    # Rolling over prior n_games meetings (shift-1 excludes current game)
    grp = hist.groupby(["focal", "opponent"])
    hist["h2h_win_rate"] = grp["focal_win"].transform(
        lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
    )
    hist["h2h_run_diff"] = grp["focal_rd"].transform(
        lambda s: s.shift(1).rolling(n_games, min_periods=1).mean()
    )

    # Pull out only the home-team perspective to merge back
    home_h2h = (
        hist.rename(columns={"focal": "home_team", "opponent": "away_team",
                              "h2h_win_rate": "h2h_home_win_rate",
                              "h2h_run_diff": "h2h_home_run_diff"})
        [["Date", "home_team", "away_team", "h2h_home_win_rate", "h2h_home_run_diff"]]
        .drop_duplicates(["Date", "home_team", "away_team"])
    )

    games = games.merge(home_h2h, on=["Date", "home_team", "away_team"], how="left")

    cov = games["h2h_home_win_rate"].notna().mean()
    print(f"  H2H coverage: {cov:.1%}  (NaN only for first-ever meetings)")
    return games


# ---------------------------------------------------------------------------
# Step 11 – weather at game time
#
# Temperature, wind speed, and wind component toward CF affect scoring rates
# directly (ball carries farther in heat/thin air, wind blowing out inflates
# offense).  We use Open-Meteo's 7 pm local hourly weather for the home park.
#
# Features:
#   temp_f          – temperature (°F) at ~7 pm local
#   wind_speed_mph  – wind speed (mph)
#   wind_to_cf      – wind component toward center field (positive = blowing out)
#   humidity_pct    – relative humidity (minor impact, included for completeness)
# ---------------------------------------------------------------------------

def attach_weather(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather observations into the games DataFrame.
    Computes wind_to_cf using park-specific CF bearing.
    Overrides wind/humidity to neutral values for full-dome parks.
    """
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"], format="mixed").dt.date
    games["_date"] = games["Date"].dt.date

    games = games.merge(
        w.rename(columns={"date": "_date"}),
        on=["_date", "home_team"],
        how="left",
    ).drop(columns=["_date"])

    # Neutral dome overrides
    for team in FULL_DOME:
        mask = games["home_team"] == team
        games.loc[mask, "temp_f"]         = 72.0
        games.loc[mask, "wind_speed_mph"] = 0.0
        games.loc[mask, "wind_dir_deg"]   = 0.0
        games.loc[mask, "humidity_pct"]   = 50.0

    # Wind component toward center field (park-specific)
    games["wind_to_cf"] = games.apply(
        lambda r: wind_to_cf(
            r["wind_speed_mph"] if pd.notna(r["wind_speed_mph"]) else 0.0,
            r["wind_dir_deg"]   if pd.notna(r["wind_dir_deg"])   else 0.0,
            r["home_team"],
        ),
        axis=1,
    )

    cov = games["temp_f"].notna().mean()
    print(f"  Weather coverage: {cov:.1%}")
    return games


# ---------------------------------------------------------------------------
# Step 12 – umpire run factor
# ---------------------------------------------------------------------------
#
# Home plate umpires vary in how they call the strike zone:
#   - A tight/small zone → more walks, more pitches per PA → more baserunners
#     → more runs scored (hitter-friendly)
#   - A wide/large zone  → more Ks, fewer walks → fewer runs (pitcher-friendly)
#
# We quantify this as: umpire's prior games' total runs/game divided by the
# year's league-average runs/game.  Values > 1 = hitter-friendly, < 1 = neutral.
#
# Leakage prevention: we use an expanding mean of past games only (shift(1)),
# requiring ≥ 20 prior games before emitting a non-NaN value.
#
# Feature: ump_run_factor  (default 1.0 when ump is unknown or has < 20 games)
# ---------------------------------------------------------------------------

def attach_umpire_factor(games: pd.DataFrame,
                         ump_logs: pd.DataFrame,
                         pitcher_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each umpire's historical run factor and attach to games.
    ump_run_factor > 1: ump tends to allow more runs than average (hitter-friendly)
    ump_run_factor < 1: ump tends to suppress runs (pitcher-friendly)
    """
    # ── 1. Total runs scored in each game (from pitcher game logs) ──────────
    # runs_allowed per pitcher = runs scored by the batting team against that pitcher
    # Summing all pitchers in a game gives total runs by both teams.
    pl = pitcher_logs.copy()
    pl["game_date"] = pd.to_datetime(pl["game_date"])
    game_runs = (pl.groupby(["game_date", "home_team"])["runs_allowed"]
                   .sum()
                   .reset_index()
                   .rename(columns={"runs_allowed": "total_runs"}))

    # ── 2. Attach total runs to ump log ────────────────────────────────────
    ul = ump_logs.copy()
    ul["game_date"] = pd.to_datetime(ul["game_date"])
    ul = ul.merge(game_runs, on=["game_date", "home_team"], how="inner")
    ul["year"] = ul["game_date"].dt.year

    # ── 3. League-average runs/game by year ─────────────────────────────────
    league_avg = ul.groupby("year")["total_runs"].mean().to_dict()

    # ── 4. Per-ump expanding mean (prior games only, min 20 games) ──────────
    ul = ul.sort_values(["ump_name", "game_date"]).reset_index(drop=True)
    ul["ump_prior_runs"] = (ul.groupby("ump_name")["total_runs"]
                              .transform(lambda x: x.shift(1).expanding(min_periods=20).mean()))

    # ── 5. Normalize by league average for that year ────────────────────────
    ul["ump_run_factor"] = ul.apply(
        lambda r: r["ump_prior_runs"] / league_avg.get(r["year"], 9.0)
        if pd.notna(r["ump_prior_runs"]) else np.nan,
        axis=1,
    )

    # ── 6. Dictionary lookup — avoids merge-induced row duplication ─────────
    # Key: (date_str, home_team) → ump_run_factor
    # Use string dates to sidestep any datetime dtype mismatches.
    ul["_date_str"] = ul["game_date"].dt.strftime("%Y-%m-%d")
    # For doubleheaders: multiple umps on same date/park — keep the first (arbitrary but consistent)
    ul_dedup = ul.drop_duplicates(subset=["_date_str", "home_team"], keep="first")
    factor_dict: dict[tuple, float] = {
        (row["_date_str"], row["home_team"]): row["ump_run_factor"]
        for _, row in ul_dedup.iterrows()
        if pd.notna(row["ump_run_factor"])
    }

    games["ump_run_factor"] = [
        factor_dict.get((pd.Timestamp(d).strftime("%Y-%m-%d"), ht), 1.0)
        for d, ht in zip(games["Date"], games["home_team"])
    ]

    cov = (games["ump_run_factor"] != 1.0).mean()
    print(f"  Umpire factor coverage (non-default): {cov:.1%}  "
          f"(NaN→1.0 for early-career umps or missing data)")
    return games


# ---------------------------------------------------------------------------
# Step 13 – IL (Injured List) counts
# ---------------------------------------------------------------------------
#
# More players on the IL → weaker effective roster → lower win probability.
# We use monthly snapshots of each team's IL count, then join each game to
# the most recent snapshot before that game date (merge_asof).
#
# Features:
#   home_il_count   – home team's IL count at game time
#   away_il_count   – away team's IL count at game time
#   il_diff         – home_il_count - away_il_count
#                     negative = home team more depleted
# ---------------------------------------------------------------------------

def attach_il_counts(games: pd.DataFrame,
                     il_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join monthly IL snapshots to games using a forward-fill merge_asof
    (each game gets the most recent snapshot taken before that game date).
    """
    il = il_df.copy()
    il["date"] = pd.to_datetime(il["date"])
    il = il.sort_values("date")

    games["_gdate"] = pd.to_datetime(games["Date"]).dt.normalize()

    def latest_il(team_col: str, out_col: str) -> pd.Series:
        team_il = il.copy()
        team_il = team_il.rename(columns={"team": "_team", "il_count": out_col})
        # For each unique team in games, merge_asof
        result = pd.merge_asof(
            games[["_gdate", team_col]].rename(columns={team_col: "_team"})
                                       .sort_values("_gdate"),
            team_il[["date", "_team", out_col]].sort_values("date"),
            left_on="_gdate",
            right_on="date",
            by="_team",
            direction="backward",   # use most recent snapshot ≤ game date
        )[out_col]
        # Restore original order
        result.index = games[["_gdate", team_col]].rename(
            columns={team_col: "_team"}).sort_values("_gdate").index
        return result.reindex(games.index)

    games["home_il_count"] = latest_il("home_team", "home_il_count")
    games["away_il_count"] = latest_il("away_team", "away_il_count")
    games["il_diff"]       = games["home_il_count"] - games["away_il_count"]
    games = games.drop(columns=["_gdate"])

    cov = games["home_il_count"].notna().mean()
    print(f"  IL count coverage: {cov:.1%}")
    return games


# ---------------------------------------------------------------------------
# Step 14 – build final feature matrix
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "home_rolling_rd",
    "away_rolling_rd",
    "rd_diff",
    "home_rolling_rs",
    "away_rolling_rs",
    "rs_diff",
    # Short-term form: 7-game rolling run differential
    "home_last7_rd",
    "away_last7_rd",
    "last7_rd_diff",
    # Momentum: 7-game minus 15-game run diff (positive = heating up)
    "home_momentum",
    "away_momentum",
    "momentum_diff",
    # Win/loss streak going into the game
    "home_streak",
    "away_streak",
    "streak_diff",
    # Season-level park-adjusted SP ERA (full season fallback)
    "home_sp_era_adj",
    "away_sp_era_adj",
    "sp_era_adj_diff",
    # In-season rolling SP ERA (last 5 starts) — higher signal
    "home_sp_inseason_era",
    "away_sp_inseason_era",
    "sp_inseason_era_diff",
    # In-season rolling bullpen RA/9 (last 15 days) — higher signal
    "home_bullpen_inseason_era",
    "away_bullpen_inseason_era",
    "bullpen_inseason_era_diff",
    "park_factor",
    # Rest & travel
    "home_days_rest",
    "away_days_rest",
    "rest_diff",
    "away_travel_miles",
    "travel_diff",
    # Head-to-head history (last 10 meetings)
    "h2h_home_win_rate",
    "h2h_home_run_diff",
    # Bullpen usage / fatigue (outs thrown in last 3 calendar days)
    "home_bullpen_outs_3d",
    "away_bullpen_outs_3d",
    "bullpen_usage_diff",
    # Prior-season team batting quality (PA-weighted OPS)
    "home_team_ops",
    "away_team_ops",
    "ops_diff",
    # Prior-season Pythagorean win% (stable team quality anchor)
    "home_prior_win_pct",
    "away_prior_win_pct",
    "prior_win_pct_diff",
    # Game-time weather at home park
    "temp_f",
    "wind_speed_mph",
    "wind_to_cf",
    "humidity_pct",
    # Umpire tendency (historical run factor vs. league average)
    "ump_run_factor",
    # Injured List counts (monthly snapshots, forward-filled to game date)
    "home_il_count",
    "away_il_count",
    "il_diff",
    # IL quality: sum of prior-year WAR for players on the IL
    "home_il_war",
    "away_il_war",
    "il_war_diff",
    # Park dimensions (static per home team)
    "park_lf_dist",
    "park_cf_dist",
    "park_rf_dist",
    "park_lf_wall_ht",
    "park_altitude_ft",
    # SP pitch stuff from FanGraphs (season-level, park-adjusted via xFIP)
    "home_sp_fbv",
    "away_sp_fbv",
    "sp_fbv_diff",
    "home_sp_swstr",
    "away_sp_swstr",
    "sp_swstr_diff",
    "home_sp_k_pct",
    "away_sp_k_pct",
    "sp_k_pct_diff",
    "home_sp_xfip",
    "away_sp_xfip",
    "sp_xfip_diff",
    # SP sample size — PA batted in current season (low = stats less reliable)
    "home_sp_pa",
    "away_sp_pa",
    "sp_pa_diff",
    # Prior-season Statcast power metrics (barrel rate, hard hit%)
    "home_barrel_pct",
    "away_barrel_pct",
    "barrel_pct_diff",
    "home_hard_hit_pct",
    "away_hard_hit_pct",
    "hard_hit_pct_diff",
    # Prediction-time-only features (NaN in training; imputed to median)
    # Vegas consensus moneyline (devigged home-win probability)
    "vegas_home_prob",
    # Confirmed lineup average OPS
    "home_lineup_ops",
    "away_lineup_ops",
    "lineup_ops_diff",
    # Batting splits vs SP handedness (OPS vs LHP / RHP)
    "home_batting_ops_vs_sp",
    "away_batting_ops_vs_sp",
    "batting_ops_vs_sp_diff",
]

TARGET_COL = "home_win"


def build_feature_matrix(games: pd.DataFrame) -> pd.DataFrame:
    games["rd_diff"]                   = games["home_rolling_rd"]           - games["away_rolling_rd"]
    games["rs_diff"]                   = games["home_rolling_rs"]           - games["away_rolling_rs"]
    games["last7_rd_diff"]             = games["home_last7_rd"]             - games["away_last7_rd"]
    games["home_momentum"]             = games["home_last7_rd"]             - games["home_rolling_rd"]
    games["away_momentum"]             = games["away_last7_rd"]             - games["away_rolling_rd"]
    games["momentum_diff"]             = games["home_momentum"]             - games["away_momentum"]
    games["streak_diff"]               = games["home_streak"]               - games["away_streak"]
    games["sp_era_adj_diff"]           = games["away_sp_era_adj"]           - games["home_sp_era_adj"]
    games["sp_inseason_era_diff"]      = games["away_sp_inseason_era"]      - games["home_sp_inseason_era"]
    games["bullpen_inseason_era_diff"] = games["away_bullpen_inseason_era"] - games["home_bullpen_inseason_era"]
    games["rest_diff"]                 = games["home_days_rest"]            - games["away_days_rest"]
    games["travel_diff"]               = games["away_travel_miles"]         - games["home_travel_miles"]
    games["bullpen_usage_diff"]        = games["away_bullpen_outs_3d"]      - games["home_bullpen_outs_3d"]
    games["ops_diff"]                  = games["home_team_ops"]             - games["away_team_ops"]
    games["prior_win_pct_diff"]        = games["home_prior_win_pct"]        - games["away_prior_win_pct"]
    # SP stuff diffs (higher FBv/K%/SwStr% + lower xFIP favors home)
    if "home_sp_fbv" in games.columns:
        games["sp_fbv_diff"]   = games["home_sp_fbv"]   - games["away_sp_fbv"]
        games["sp_swstr_diff"] = games["home_sp_swstr"] - games["away_sp_swstr"]
        games["sp_k_pct_diff"] = games["home_sp_k_pct"] - games["away_sp_k_pct"]
        games["sp_xfip_diff"]  = games["away_sp_xfip"]  - games["home_sp_xfip"]
        games["sp_pa_diff"]    = games["home_sp_pa"]    - games["away_sp_pa"]
    # Prior-season Statcast power metrics
    if "home_barrel_pct" in games.columns:
        games["barrel_pct_diff"]   = games["home_barrel_pct"]   - games["away_barrel_pct"]
        games["hard_hit_pct_diff"] = games["home_hard_hit_pct"] - games["away_hard_hit_pct"]
    # IL WAR diff (positive = home team has more WAR on IL = more depleted)
    if "home_il_war" in games.columns:
        games["il_war_diff"] = games["home_il_war"] - games["away_il_war"]
    else:
        for col in ["home_il_war", "away_il_war", "il_war_diff"]:
            games[col] = np.nan

    # Prediction-time-only (will be NaN in training data; imputed to median)
    for col in ["vegas_home_prob", "home_lineup_ops", "away_lineup_ops",
                "home_batting_ops_vs_sp", "away_batting_ops_vs_sp"]:
        if col not in games.columns:
            games[col] = np.nan
    games["lineup_ops_diff"]          = games["home_lineup_ops"]          - games["away_lineup_ops"]
    games["batting_ops_vs_sp_diff"]   = games["home_batting_ops_vs_sp"]   - games["away_batting_ops_vs_sp"]

    available = [c for c in FEATURE_COLS if c in games.columns]
    missing = [c for c in FEATURE_COLS if c not in games.columns]
    coverage_pct = len(available) / len(FEATURE_COLS) * 100
    if missing:
        print(f"  WARNING: {len(missing)}/{len(FEATURE_COLS)} feature cols absent "
              f"({coverage_pct:.0f}% coverage): {missing[:10]}{'…' if len(missing) > 10 else ''}")

    out = games[["Date", "year", "home_team", "away_team", TARGET_COL] + available].copy()
    # Drop rows where either rolling_rd is NaN — these are early-season games
    # with < min_periods prior games and would bias training with noisy inputs.
    out = out.dropna(subset=["home_rolling_rd", "away_rolling_rd"], how="any")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sp_path = os.path.join(DATA_DIR, "game_sp.csv")
    if not os.path.exists(sp_path):
        raise FileNotFoundError(
            "data/game_sp.csv not found — run fetch_sp_data.py first."
        )

    gl_path  = os.path.join(DATA_DIR, "pitcher_game_logs.csv")
    ump_path = os.path.join(DATA_DIR, "umpire_game_logs.csv")
    if not os.path.exists(gl_path):
        raise FileNotFoundError(
            "data/pitcher_game_logs.csv not found — run parse_retrosheet_events.py first."
        )

    batting_path = os.path.join(DATA_DIR, "batting_stats.csv")
    weather_path = os.path.join(DATA_DIR, "weather.csv")
    if not os.path.exists(batting_path):
        raise FileNotFoundError(
            "data/batting_stats.csv not found — run data_ingestion.py first."
        )
    if not os.path.exists(weather_path):
        raise FileNotFoundError(
            "data/weather.csv not found — run fetch_weather.py first."
        )

    print("Loading raw data...")
    raw_logs      = pd.read_csv(os.path.join(DATA_DIR, "game_logs_raw.csv"))
    pitcher_stats = pd.read_csv(os.path.join(DATA_DIR, "pitcher_stats.csv"))
    batting_stats = pd.read_csv(batting_path)
    weather       = pd.read_csv(weather_path, parse_dates=["date"])
    game_sp       = pd.read_csv(sp_path, parse_dates=["Date"])
    game_logs     = pd.read_csv(gl_path,  parse_dates=["game_date"])
    ump_logs      = pd.read_csv(ump_path, parse_dates=["game_date"]) if os.path.exists(ump_path) else None
    il_path       = os.path.join(DATA_DIR, "il_counts.csv")
    il_df         = pd.read_csv(il_path,  parse_dates=["date"]) if os.path.exists(il_path) else None

    print("Cleaning game logs...")
    games = clean_game_logs(raw_logs)
    print(f"  {len(games):,} home games from pybaseball cache")

    # Merge daily-updated live results (written by daily_update.py)
    live_path = os.path.join(DATA_DIR, "game_logs_live.csv")
    if os.path.exists(live_path):
        live = pd.read_csv(live_path, parse_dates=["Date"])
        existing_keys = set(zip(games["Date"].astype(str), games["home_team"]))
        new_rows = live[
            ~live.apply(lambda r: (str(r["Date"])[:10], r["home_team"]) in existing_keys, axis=1)
        ]
        if not new_rows.empty:
            games = pd.concat([games, new_rows], ignore_index=True).sort_values("Date")
            print(f"  +{len(new_rows)} games from game_logs_live.csv  → {len(games):,} total")
        else:
            print(f"  game_logs_live.csv: no new games beyond pybaseball cache")
    print(f"  {len(games):,} home games total")

    print("Adding rolling run-differential + runs scored...")
    games = add_rolling_stats(games)

    print("Computing park factors...")
    games = attach_park_factor(games)
    park_factors = games[["home_team", "year", "park_factor"]].drop_duplicates()

    print("Building pitcher ERA lookup (park-adjusted)...")
    pitcher_era       = build_pitcher_era_lookup(pitcher_stats, park_factors)
    team_era_fallback = build_team_era_fallback(pitcher_stats, park_factors)

    print("Attaching per-game starting pitcher ERA (park-adjusted)...")
    games = attach_sp_era(games, game_sp, pitcher_era, team_era_fallback)

    print("Attaching park dimensions...")
    games = attach_park_dimensions(games)

    print("Attaching SP pitch stuff (FanGraphs)...")
    stuff_path = os.path.join(DATA_DIR, "pitcher_stuff.csv")
    if os.path.exists(stuff_path):
        pitcher_stuff = pd.read_csv(stuff_path)
        games = attach_sp_stuff(games, pitcher_stuff)
    else:
        print("  pitcher_stuff.csv not found — run fetch_pitcher_stuff.py first")
        games = attach_sp_stuff(games, None)

    print("Computing rest days and travel distance...")
    games = compute_rest_and_travel(games)

    print("Attaching prior-season Statcast batting (barrel rate, hard hit%)...")
    statcast_path = os.path.join(DATA_DIR, "statcast_batting.csv")
    if os.path.exists(statcast_path):
        statcast_df = pd.read_csv(statcast_path)
        games = attach_statcast_batting(games, statcast_df)
    else:
        print("  statcast_batting.csv not found — run fetch_statcast_batting.py first")
        games = attach_statcast_batting(games, None)

    print("Attaching prior-season batting quality (team OPS)...")
    games = attach_batting_quality(games, batting_stats)

    print("Attaching prior-season Pythagorean win%...")
    games = attach_prior_win_pct(games)

    print("Computing head-to-head history...")
    games = compute_h2h_stats(games)

    print("Attaching weather data...")
    games = attach_weather(games, weather)

    print("Attaching umpire run factor...")
    if ump_logs is not None:
        games = attach_umpire_factor(games, ump_logs, game_logs)
    else:
        print("  umpire_game_logs.csv not found — run parse_retrosheet_events.py; defaulting to 1.0")
        games["ump_run_factor"] = 1.0

    print("Attaching IL (Injured List) counts...")
    if il_df is not None:
        games = attach_il_counts(games, il_df)
    else:
        print("  il_counts.csv not found — run fetch_il_data.py; defaulting to 0")
        games["home_il_count"] = 0
        games["away_il_count"] = 0
        games["il_diff"]       = 0

    print("Attaching IL quality (WAR-weighted)...")
    il_quality_path = os.path.join(DATA_DIR, "il_quality.csv")
    if os.path.exists(il_quality_path):
        il_quality_df = pd.read_csv(il_quality_path, parse_dates=["date"])

        def _attach_il_war(team_col: str, out_col: str) -> pd.Series:
            tq = il_quality_df.rename(columns={"team": "_team", "il_war_score": out_col})
            result = pd.merge_asof(
                games[["_gdate", team_col]].rename(columns={team_col: "_team"})
                                           .sort_values("_gdate"),
                tq[["date", "_team", out_col]].sort_values("date"),
                left_on="_gdate", right_on="date",
                by="_team", direction="backward",
            )[out_col]
            result.index = games[["_gdate", team_col]].rename(
                columns={team_col: "_team"}).sort_values("_gdate").index
            return result.reindex(games.index)

        games["_gdate"] = pd.to_datetime(games["Date"]).dt.normalize()
        games["home_il_war"] = _attach_il_war("home_team", "home_il_war")
        games["away_il_war"] = _attach_il_war("away_team", "away_il_war")
        games = games.drop(columns=["_gdate"])
        cov = games["home_il_war"].notna().mean()
        print(f"  IL quality coverage: {cov:.1%}")
    else:
        print("  il_quality.csv not found — run fetch_il_data.py; defaulting to NaN")
        games["home_il_war"] = np.nan
        games["away_il_war"] = np.nan

    print("Attaching in-season rolling SP + bullpen ERA...")
    games = attach_inseason_stats(games, game_logs, game_sp, park_factors)

    print("Attaching historical Vegas consensus odds...")

    def _devig(hml, aml):
        def imp(ml):
            ml = float(ml)
            return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)
        ph, pa = imp(hml), imp(aml)
        return ph / (ph + pa) if (ph + pa) > 0 else np.nan

    odds_frames = []

    # Action Network files (game_date + consensus_prob already computed)
    for odds_file in ["action_network_odds_2026.csv", "action_network_odds_2022_2025.csv"]:
        p = os.path.join(DATA_DIR, odds_file)
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={"game_date": str})
            df = df.rename(columns={"game_date": "Date", "consensus_prob": "vegas_home_prob"})
            odds_frames.append(df[["Date", "home_team", "away_team", "vegas_home_prob"]])

    # historical_odds.csv (2015-2021): has 'date' + raw home_ml/away_ml, no consensus_prob
    hist_path = os.path.join(DATA_DIR, "historical_odds.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        hist = hist.rename(columns={"date": "Date"})
        hist["vegas_home_prob"] = hist.apply(
            lambda r: _devig(r["home_ml"], r["away_ml"])
            if pd.notna(r.get("home_ml")) and pd.notna(r.get("away_ml")) else np.nan,
            axis=1,
        )
        odds_frames.append(hist[["Date", "home_team", "away_team", "vegas_home_prob"]])

    if odds_frames:
        odds_df = (pd.concat(odds_frames, ignore_index=True)
                     .dropna(subset=["vegas_home_prob"]))
        odds_df["Date"] = pd.to_datetime(odds_df["Date"])
        odds_df = odds_df.drop_duplicates(["Date", "home_team", "away_team"])
        games = games.merge(
            odds_df[["Date", "home_team", "away_team", "vegas_home_prob"]],
            on=["Date", "home_team", "away_team"], how="left"
        )
        coverage = games["vegas_home_prob"].notna().mean()
        n_games  = games["vegas_home_prob"].notna().sum()
        print(f"  Vegas odds coverage: {coverage:.1%} of games ({n_games:,} games)")
    else:
        print("  No odds files found — run fetch_historical_odds.py first")
        games["vegas_home_prob"] = np.nan

    print("Building feature matrix...")
    features = build_feature_matrix(games)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Columns: {features.columns.tolist()}")
    print(f"  Home win rate: {features[TARGET_COL].mean():.3f}")

    out_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} — run train_model.py next.")
