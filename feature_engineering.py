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

import json
import math
import os
import re
import unicodedata
import pandas as pd
import numpy as np

# Import weather helpers (park CF bearings + wind projection)
from fetch_weather import PARK_CF_BEARING, FULL_DOME, wind_to_cf
from fetch_pitcher_stuff import get_pitcher_stuff
from feature_defaults import apply_feature_defaults

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
    "oakland athletics": "ATH", "athletics": "ATH",
    "philadelphia phillies": "PHI",
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
    "oakland": "ATH", "philadelphia": "PHI", "pittsburgh": "PIT",
    "san diego": "SDP", "seattle": "SEA", "san francisco": "SFG",
    "st. louis": "STL", "tampa bay": "TBR", "texas": "TEX",
    "toronto": "TOR", "washington": "WSN",
}

SCHEDULE_ABBREV_MAP = {
    "CWS": "CHW", "KC": "KCR", "SD": "SDP", "SF": "SFG",
    "TB": "TBR", "WSH": "WSN", "OAK": "ATH",
}


def normalize_schedule_team(name: str) -> str:
    s = str(name).strip().upper()
    return SCHEDULE_ABBREV_MAP.get(s, s)


def normalize_bref_team(name: str) -> str:
    s = str(name).strip().lower()
    return BREF_NAME_TO_ABBREV.get(s, s.upper())


def add_game_identity(games: pd.DataFrame,
                      date_col: str = "Date",
                      home_col: str = "home_team",
                      away_col: str = "away_team") -> pd.DataFrame:
    """
    Add doubleheader-safe per-game identity columns.

    game_number is the sequence within a same-date/same-teams doubleheader.
    game_id is a stable synthetic key used when source files do not share MLB's
    game_pk or Retrosheet game_id.
    """
    out = games.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if "_source_order" not in out.columns:
        out["_source_order"] = np.arange(len(out))
    out = out.sort_values([date_col, home_col, away_col, "_source_order"]).copy()
    out["game_number"] = (
        out.groupby([date_col, home_col, away_col], dropna=False)
        .cumcount()
        .add(1)
        .astype(int)
    )
    out["game_id"] = (
        out[date_col].dt.strftime("%Y%m%d")
        + "_"
        + out[home_col].astype(str)
        + "_"
        + out[away_col].astype(str)
        + "_"
        + out["game_number"].astype(str)
    )
    return out.drop(columns=["_source_order"], errors="ignore")


def game_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    keys = ["Date", "home_team", "away_team"]
    if "game_number" in left.columns and "game_number" in right.columns:
        keys.append("game_number")
    return keys


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
    "ATH": (37.7516, -122.2005),   # Athletics historical home park proxy
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
    "ATH": {"lf_dist": 330, "cf_dist": 400, "rf_dist": 330, "lf_wall_ht":  8.0, "altitude_ft":   25},
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
    df["_source_order"] = np.arange(len(df))

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

    out = df[["Date", "year", "team", "Opp", "home_win", "R", "RA", "_source_order"]].rename(columns={
        "team": "home_team",
        "Opp":  "away_team",
        "R":    "home_runs",
        "RA":   "away_runs",
    })
    return add_game_identity(out)


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
    home = games[["game_id", "Date", "home_team", "home_runs", "away_runs"]].copy()
    home.columns = ["game_id", "Date", "team", "runs_for", "runs_against"]
    home["venue"] = "home"

    away = games[["game_id", "Date", "away_team", "away_runs", "home_runs"]].copy()
    away.columns = ["game_id", "Date", "team", "runs_for", "runs_against"]
    away["venue"] = "away"

    tg = pd.concat([home, away]).sort_values(["team", "Date", "game_id"]).reset_index(drop=True)
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

    # Current-season running win% (shift-1, reset each calendar year)
    tg["win"] = (tg["run_diff"] > 0).astype(float)
    tg["_year"] = tg["Date"].dt.year
    tg["season_win_pct"] = (
        tg.groupby(["team", "_year"])["win"]
        .transform(lambda s: s.shift(1).expanding(min_periods=10).mean())
    )
    tg = tg.drop(columns=["win", "_year"])

    # Home-venue and away-venue rolling run diff (shift-1 within each venue subset)
    for venue_tag in ("home", "away"):
        mask = tg["venue"] == venue_tag
        sub  = tg[mask].copy()
        sub[f"rolling_{venue_tag}_rd"] = sub.groupby("team")["run_diff"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).mean()
        )
        tg = tg.merge(
            sub[["team", "game_id", f"rolling_{venue_tag}_rd"]],
            on=["team", "game_id"], how="left"
        )

    def merge_side(side_col, rd_col, rs_col, rd7_col, streak_col, swp_col,
                   h_rd_col, a_rd_col):
        side = tg.merge(
            games[["game_id", side_col]],
            left_on=["game_id", "team"], right_on=["game_id", side_col],
            how="inner"
        )[[side_col, "game_id", "rolling_run_diff", "rolling_runs_scored",
           "last7_run_diff", "streak", "season_win_pct",
           "rolling_home_rd", "rolling_away_rd"]].rename(columns={
            "rolling_run_diff":    rd_col,
            "rolling_runs_scored": rs_col,
            "last7_run_diff":      rd7_col,
            "streak":              streak_col,
            "season_win_pct":      swp_col,
            "rolling_home_rd":     h_rd_col,
            "rolling_away_rd":     a_rd_col,
        }).drop_duplicates(["game_id", side_col])
        return side

    home_stats = merge_side("home_team", "home_rolling_rd", "home_rolling_rs",
                             "home_last7_rd", "home_streak", "home_season_win_pct",
                             "home_home_rd", "home_away_rd")
    away_stats = merge_side("away_team", "away_rolling_rd", "away_rolling_rs",
                             "away_last7_rd", "away_streak", "away_season_win_pct",
                             "away_home_rd", "away_away_rd")

    games = games.merge(home_stats, on=["game_id", "home_team"], how="left")
    games = games.merge(away_stats, on=["game_id", "away_team"], how="left")
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
    Join Retrosheet SP assignments to each game, then look up prior-season raw
    and park-adjusted ERA. Falls back to prior-season team ERA when SP name is
    not found.
    """
    gs = game_sp.copy()
    gs["Date"] = pd.to_datetime(gs["Date"])
    gs["home_team"] = gs["home_team"].apply(normalize_schedule_team)
    gs["away_team"] = gs["away_team"].apply(normalize_schedule_team)
    gs = add_game_identity(gs)
    gs["home_sp_norm"] = gs["home_sp_name"].apply(_normalize_sp_name)
    gs["away_sp_norm"] = gs["away_sp_name"].apply(_normalize_sp_name)

    merge_keys = game_merge_keys(games, gs)
    games = games.merge(
        gs[merge_keys + ["home_sp_norm", "away_sp_norm"]],
        on=merge_keys, how="left"
    )

    era_raw = pitcher_era.set_index(["name_norm", "year"])["ERA"]
    era_adj = pitcher_era.set_index(["name_norm", "year"])["ERA_adj"]

    def lookup(row, sp_col, lookup_series):
        name = row.get(sp_col)
        year = row.get("year")
        if pd.isna(name) or pd.isna(year):
            return np.nan
        return lookup_series.get((name, int(year) - 1), np.nan)

    games["home_sp_era"]     = games.apply(lookup, sp_col="home_sp_norm", lookup_series=era_raw, axis=1)
    games["away_sp_era"]     = games.apply(lookup, sp_col="away_sp_norm", lookup_series=era_raw, axis=1)
    games["home_sp_era_adj"] = games.apply(lookup, sp_col="home_sp_norm", lookup_series=era_adj, axis=1)
    games["away_sp_era_adj"] = games.apply(lookup, sp_col="away_sp_norm", lookup_series=era_adj, axis=1)

    # Fallback to team ERA
    fb = team_era_fallback.copy()
    fb["year"] = fb["year"] + 1
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
                       era_adj.get((r["home_sp_norm"], int(r["year"]) - 1)
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
# Joined at the prior-season level for historical training: for game in year Y,
# use the SP's FanGraphs stats from year Y-1. We avoid current-season xFIP,
# FBv, SwStr%, and K% here because pitcher_stuff.csv stores season-level rows,
# not as-of-game snapshots, so current-year values would leak future games.
# Throws is stored separately so predict.py can look up batting splits.
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
    hand_cache_path = os.path.join(DATA_DIR, "pitcher_handedness.json")
    if os.path.exists(hand_cache_path):
        with open(hand_cache_path) as f:
            hand_cache = json.load(f)
    else:
        hand_cache = {}

    def _stuff(name_norm, team, year):
        row = get_pitcher_stuff(
            name_norm or "",
            team,
            int(year) - 1,
            stuff,
        )
        if not row.get("Throws") and name_norm:
            row["Throws"] = hand_cache.get(name_norm)
        return row

    for side, sp_col, team_col in [("home", "home_sp_norm", "home_team"),
                                    ("away", "away_sp_norm", "away_team")]:
        yr   = games["year"]
        name = games[sp_col] if sp_col in games.columns else pd.Series([None]*len(games))
        team = games[team_col]
        rows = [_stuff(n, t, y) for n, t, y in zip(name, team, yr)]

        games[f"{side}_sp_fbv"]    = [r.get("FBv", np.nan) for r in rows]
        games[f"{side}_sp_swstr"]  = [r.get("SwStr_pct", np.nan) for r in rows]
        games[f"{side}_sp_k_pct"]  = [r.get("K_pct", np.nan) for r in rows]
        games[f"{side}_sp_xfip"]   = [r.get("xFIP", np.nan) for r in rows]
        games[f"{side}_sp_pa"]     = [r.get("pa", np.nan) for r in rows]
        games[f"{side}_sp_throws"] = [r.get("Throws") for r in rows]

    cov = games["home_sp_fbv"].notna().mean()
    print(f"  SP stuff coverage: {cov:.1%}  (prior-season, no leakage)")
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


def compute_sp_workload(game_logs: pd.DataFrame,
                         game_sp: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute the starting pitcher's:
      - sp_days_rest  : days since their previous start (capped 0–10; NaN for debut)
      - sp_outs_last  : outs recorded in their previous start (proxy for pitch count)

    Both use shift-1 per pitcher so the current game is always excluded.
    Joined back via game_sp pitcher IDs → returns one row per game with
    home_sp_days_rest, away_sp_days_rest, home_sp_outs_last, away_sp_outs_last.
    """
    starters = game_logs[game_logs["is_starter"] == 1].copy()
    starters["game_date"] = pd.to_datetime(starters["game_date"])
    starters = starters.sort_values(["pitcher_id", "game_date"])

    grp = starters.groupby("pitcher_id")
    starters["prev_start_date"] = grp["game_date"].transform(lambda s: s.shift(1))
    starters["sp_outs_last"]    = grp["outs_recorded"].transform(lambda s: s.shift(1))
    starters["sp_days_rest"]    = (
        (starters["game_date"] - starters["prev_start_date"]).dt.days - 1
    ).clip(0, 10)

    gs = game_sp.copy()
    gs["Date"] = pd.to_datetime(gs["Date"])
    gs["home_team"] = gs["home_team"].apply(normalize_schedule_team)
    gs["away_team"] = gs["away_team"].apply(normalize_schedule_team)
    gs = add_game_identity(gs)

    work = starters[["game_date", "pitcher_id", "sp_days_rest", "sp_outs_last"]]

    result = gs.merge(
        work.rename(columns={"game_date": "Date",
                              "sp_days_rest": "home_sp_days_rest",
                              "sp_outs_last": "home_sp_outs_last"}),
        left_on=["Date", "home_sp_id"], right_on=["Date", "pitcher_id"], how="left"
    ).drop(columns=["pitcher_id"])

    result = result.merge(
        work.rename(columns={"game_date": "Date",
                              "sp_days_rest": "away_sp_days_rest",
                              "sp_outs_last": "away_sp_outs_last"}),
        left_on=["Date", "away_sp_id"], right_on=["Date", "pitcher_id"], how="left"
    ).drop(columns=["pitcher_id"])

    cov = result["home_sp_days_rest"].notna().mean()
    print(f"  SP workload coverage: {cov:.1%}  (days rest + outs last start)")
    cols = ["Date", "home_team", "away_team", "game_number",
                   "home_sp_days_rest", "away_sp_days_rest",
                   "home_sp_outs_last", "away_sp_outs_last"]
    return result[[c for c in cols if c in result.columns]]


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
    starters = game_logs[game_logs["is_starter"] == 1].copy()
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
    # Last-3-starts window (more recent form signal)
    starters["roll_runs_3"] = grp["runs_allowed"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).sum()
    )
    starters["roll_outs_3"] = grp["outs_recorded"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).sum()
    )
    starters["sp_inseason_ra9"]   = starters.apply(lambda r: _ra9(r["roll_runs"],   r["roll_outs"]),   axis=1)
    starters["sp_last3_ra9"]      = starters.apply(lambda r: _ra9(r["roll_runs_3"], r["roll_outs_3"]), axis=1)

    # Park-adjust both windows
    pf_lookup = park_factors.set_index(["home_team", "year"])["park_factor"]

    def park_adj_sp(row):
        pitcher_team = row["home_team"] if row["team_side"] == 1 else row["away_team"]
        year = row["game_date"].year
        pf   = pf_lookup.get((pitcher_team, year), np.nan)
        return (
            _park_adjust_era(row["sp_inseason_ra9"], pf),
            _park_adjust_era(row["sp_last3_ra9"],    pf),
        )

    adj = starters.apply(park_adj_sp, axis=1, result_type="expand")
    starters["sp_inseason_era_adj"] = adj[0]
    starters["sp_last3_era_adj"]    = adj[1]

    # Merge SP assignments to get pitcher_id per game
    gs = game_sp.copy()
    gs["Date"] = pd.to_datetime(gs["Date"])
    gs["home_team"] = gs["home_team"].apply(normalize_schedule_team)
    gs["away_team"] = gs["away_team"].apply(normalize_schedule_team)
    gs = add_game_identity(gs)

    # Join home SP stats
    home_sp = starters[["game_date", "pitcher_id",
                         "sp_inseason_era_adj", "sp_last3_era_adj"]].rename(
        columns={"game_date": "Date",
                 "sp_inseason_era_adj": "home_sp_inseason_era",
                 "sp_last3_era_adj":    "home_sp_last3_era"}
    )
    result = gs.merge(
        home_sp,
        left_on=["Date", "home_sp_id"],
        right_on=["Date", "pitcher_id"],
        how="left"
    ).drop(columns=["pitcher_id"])

    # Join away SP stats
    away_sp = starters[["game_date", "pitcher_id",
                         "sp_inseason_era_adj", "sp_last3_era_adj"]].rename(
        columns={"game_date": "Date",
                 "sp_inseason_era_adj": "away_sp_inseason_era",
                 "sp_last3_era_adj":    "away_sp_last3_era"}
    )
    result = result.merge(
        away_sp,
        left_on=["Date", "away_sp_id"],
        right_on=["Date", "pitcher_id"],
        how="left"
    ).drop(columns=["pitcher_id"])

    coverage = result["home_sp_inseason_era"].notna().mean()
    print(f"  SP in-season ERA coverage  : {coverage:.1%}")
    cols = ["Date", "home_team", "away_team", "game_number",
                   "home_sp_inseason_era", "away_sp_inseason_era",
                   "home_sp_last3_era",    "away_sp_last3_era"]
    return result[[c for c in cols if c in result.columns]]


def compute_inseason_bullpen_era(game_logs: pd.DataFrame,
                                  park_factors: pd.DataFrame,
                                  window_days: int = 15) -> pd.DataFrame:
    """
    For each (team, game_date), compute the bullpen's rolling RA/9 over
    the prior `window_days` days using only relief appearances.
    shift-1 on game_date so current game is excluded.

    Returns: DataFrame with game_date, team, bullpen_inseason_era
    """
    relievers = game_logs[game_logs["is_starter"] != 1].copy()
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
    relievers = game_logs[game_logs["is_starter"] != 1].copy()
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
    """Attach in-season rolling SP ERA, bullpen ERA, bullpen usage, and SP workload to games."""

    # --- SP ERA ---
    sp_stats = compute_inseason_sp_era(game_logs, game_sp, park_factors)
    games = games.merge(sp_stats, on=game_merge_keys(games, sp_stats), how="left")

    # --- SP workload (days rest + outs last start) ---
    sp_work = compute_sp_workload(game_logs, game_sp)
    games = games.merge(sp_work, on=game_merge_keys(games, sp_work), how="left")

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


def _weighted_split_ops(row: pd.Series) -> float:
    pa_l = float(row.get("pa_vs_lhp", 0) or 0)
    pa_r = float(row.get("pa_vs_rhp", 0) or 0)
    ops_l = float(row.get("ops_vs_lhp", np.nan))
    ops_r = float(row.get("ops_vs_rhp", np.nan))
    total = pa_l + pa_r
    if total > 0 and pd.notna(ops_l) and pd.notna(ops_r):
        return (ops_l * pa_l + ops_r * pa_r) / total
    return np.nanmean([ops_l, ops_r])


def attach_batting_splits_vs_sp(games: pd.DataFrame,
                                splits_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Attach prior-season team OPS against the opposing starter's throwing hand.
    For a game in year Y, use team split stats from Y-1 to avoid leakage.
    """
    split_cols = [
        "home_batting_ops_vs_sp", "away_batting_ops_vs_sp",
        "home_platoon_advantage", "away_platoon_advantage",
        "home_opp_sp_is_lhp", "away_opp_sp_is_lhp",
        "home_sp_is_lhp", "away_sp_is_lhp", "both_sp_same_hand",
    ]
    if splits_df is None or splits_df.empty:
        for col in split_cols:
            games[col] = np.nan
        print("  Batting splits vs SP coverage: 0.0%  (team_splits.csv missing)")
        return games

    splits = splits_df.copy()
    for col in ["ops_vs_lhp", "ops_vs_rhp", "pa_vs_lhp", "pa_vs_rhp"]:
        splits[col] = pd.to_numeric(splits[col], errors="coerce")
    splits["year"] = pd.to_numeric(splits["year"], errors="coerce").astype("Int64")
    splits["overall_split_ops"] = splits.apply(_weighted_split_ops, axis=1)
    split_lk = splits.set_index(["team", "year"]).to_dict("index")

    def _ops_for(team: str, year: int, opp_hand: str | None) -> float:
        row = split_lk.get((team, int(year) - 1))
        if not row:
            return np.nan
        hand = str(opp_hand).strip().upper() if pd.notna(opp_hand) else ""
        if hand == "L":
            return float(row.get("ops_vs_lhp", np.nan))
        if hand == "R":
            return float(row.get("ops_vs_rhp", np.nan))
        return float(row.get("overall_split_ops", np.nan))

    games["home_batting_ops_vs_sp"] = [
        _ops_for(team, year, hand)
        for team, year, hand in zip(games["home_team"], games["year"], games.get("away_sp_throws"))
    ]
    games["away_batting_ops_vs_sp"] = [
        _ops_for(team, year, hand)
        for team, year, hand in zip(games["away_team"], games["year"], games.get("home_sp_throws"))
    ]
    games["home_platoon_advantage"] = games["home_batting_ops_vs_sp"] - games["home_team_ops"]
    games["away_platoon_advantage"] = games["away_batting_ops_vs_sp"] - games["away_team_ops"]

    home_known = games["home_sp_throws"].isin(["L", "R"])
    away_known = games["away_sp_throws"].isin(["L", "R"])
    games["home_opp_sp_is_lhp"] = np.where(
        away_known, (games["away_sp_throws"] == "L").astype(float), np.nan
    )
    games["away_opp_sp_is_lhp"] = np.where(
        home_known, (games["home_sp_throws"] == "L").astype(float), np.nan
    )
    games["home_sp_is_lhp"] = np.where(
        home_known, (games["home_sp_throws"] == "L").astype(float), np.nan
    )
    games["away_sp_is_lhp"] = np.where(
        away_known, (games["away_sp_throws"] == "L").astype(float), np.nan
    )
    known_hands = games["home_sp_throws"].isin(["L", "R"]) & games["away_sp_throws"].isin(["L", "R"])
    games["both_sp_same_hand"] = np.where(
        known_hands,
        (games["home_sp_throws"] == games["away_sp_throws"]).astype(float),
        np.nan,
    )

    cov = games["home_batting_ops_vs_sp"].notna().mean()
    print(f"  Batting splits vs SP coverage: {cov:.1%}")
    return games


def attach_historical_lineups(games: pd.DataFrame,
                              lineups_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Attach archived starting-lineup split OPS from data/historical_lineups.csv.
    These features are only trainable once enough seasons have been backfilled.
    """
    cols = [
        "home_lineup_ops_vs_sp", "away_lineup_ops_vs_sp",
        "lineup_ops_vs_sp_diff",
        "home_lineup_ops_vs_lhp", "home_lineup_ops_vs_rhp",
        "away_lineup_ops_vs_lhp", "away_lineup_ops_vs_rhp",
        "home_lineup_known_batters", "away_lineup_known_batters",
    ]
    if lineups_df is None or lineups_df.empty:
        for col in cols:
            games[col] = np.nan
        print("  Historical lineup coverage: 0.0%")
        return games

    lineup = lineups_df.copy()
    lineup["game_date"] = pd.to_datetime(lineup["game_date"])
    for col in [
        "home_lineup_blend_ops_vs_lhp", "home_lineup_blend_ops_vs_rhp",
        "away_lineup_blend_ops_vs_lhp", "away_lineup_blend_ops_vs_rhp",
        "home_lineup_ops_vs_lhp", "home_lineup_ops_vs_rhp",
        "away_lineup_ops_vs_lhp", "away_lineup_ops_vs_rhp",
        "home_lineup_known_batters", "away_lineup_known_batters",
    ]:
        if col in lineup.columns:
            lineup[col] = pd.to_numeric(lineup[col], errors="coerce")

    lineup["home_team"] = lineup["home_team"].apply(normalize_schedule_team)
    lineup["away_team"] = lineup["away_team"].apply(normalize_schedule_team)
    lineup = lineup.sort_values(["game_date", "home_team", "away_team", "game_pk"])
    lineup["game_number"] = (
        lineup.groupby(["game_date", "home_team", "away_team"], dropna=False)
        .cumcount()
        .add(1)
        .astype(int)
    )

    keep = [
        "game_pk", "game_date", "home_team", "away_team", "game_number",
        "home_lineup_blend_ops_vs_lhp", "home_lineup_blend_ops_vs_rhp",
        "away_lineup_blend_ops_vs_lhp", "away_lineup_blend_ops_vs_rhp",
        "home_lineup_ops_vs_lhp", "home_lineup_ops_vs_rhp",
        "away_lineup_ops_vs_lhp", "away_lineup_ops_vs_rhp",
        "home_lineup_known_batters", "away_lineup_known_batters",
    ]
    lineup = lineup[[c for c in keep if c in lineup.columns]].drop_duplicates(["game_pk"], keep="last")

    out = games.copy()
    out["game_date"] = pd.to_datetime(out["Date"])
    merge_keys = ["game_date", "home_team", "away_team"]
    if "game_number" in out.columns and "game_number" in lineup.columns:
        merge_keys.append("game_number")
    out = out.merge(lineup, on=merge_keys, how="left")

    for side in ["home", "away"]:
        for suffix in ["lhp", "rhp"]:
            blend_col = f"{side}_lineup_blend_ops_vs_{suffix}"
            prior_col = f"{side}_lineup_ops_vs_{suffix}"
            if blend_col in out.columns:
                out[prior_col] = out[blend_col].combine_first(out[prior_col])

    out["home_lineup_ops_vs_sp"] = np.select(
        [out["away_sp_throws"].eq("L"), out["away_sp_throws"].eq("R")],
        [out["home_lineup_ops_vs_lhp"], out["home_lineup_ops_vs_rhp"]],
        default=np.nan,
    )
    out["away_lineup_ops_vs_sp"] = np.select(
        [out["home_sp_throws"].eq("L"), out["home_sp_throws"].eq("R")],
        [out["away_lineup_ops_vs_lhp"], out["away_lineup_ops_vs_rhp"]],
        default=np.nan,
    )
    out["lineup_ops_vs_sp_diff"] = out["home_lineup_ops_vs_sp"] - out["away_lineup_ops_vs_sp"]
    # Keep LHP/RHP split columns (blend version takes priority if available)
    for side in ["home", "away"]:
        for hand in ["lhp", "rhp"]:
            blend_col = f"{side}_lineup_blend_ops_vs_{hand}"
            split_col = f"{side}_lineup_ops_vs_{hand}"
            if blend_col in out.columns:
                out[split_col] = out[blend_col].combine_first(out.get(split_col, pd.Series(dtype=float)))
    out = out.drop(columns=["game_date"])

    cov = out["home_lineup_ops_vs_sp"].notna().mean()
    print(f"  Historical lineup coverage: {cov:.1%}")
    return out


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
# min_periods=3 avoids letting one or two prior games create extreme matchup
# signals; NaN is imputed downstream until there is enough matchup history.
# ---------------------------------------------------------------------------

def compute_h2h_stats(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """
    Attach rolling head-to-head win rate and run differential to every game.
    """
    games = games.sort_values("Date").reset_index(drop=True)

    # Build a symmetric view: one row per (focal_team, opponent, game)
    home_view = games[["game_id", "Date", "home_team", "away_team", "home_win",
                        "home_runs", "away_runs"]].copy()
    home_view["focal"]     = home_view["home_team"]
    home_view["opponent"]  = home_view["away_team"]
    home_view["focal_win"] = home_view["home_win"]
    home_view["focal_rd"]  = home_view["home_runs"] - home_view["away_runs"]

    away_view = games[["game_id", "Date", "home_team", "away_team", "home_win",
                        "home_runs", "away_runs"]].copy()
    away_view["focal"]     = away_view["away_team"]
    away_view["opponent"]  = away_view["home_team"]
    away_view["focal_win"] = 1 - away_view["home_win"]
    away_view["focal_rd"]  = away_view["away_runs"] - away_view["home_runs"]

    hist = (
        pd.concat(
            [home_view[["game_id", "Date", "focal", "opponent", "focal_win", "focal_rd"]],
             away_view[["game_id", "Date", "focal", "opponent", "focal_win", "focal_rd"]]],
            ignore_index=True,
        )
        .sort_values(["focal", "opponent", "Date"])
        .reset_index(drop=True)
    )

    # Rolling over prior n_games meetings (shift-1 excludes current game)
    grp = hist.groupby(["focal", "opponent"])
    hist["h2h_win_rate"] = grp["focal_win"].transform(
        lambda s: s.shift(1).rolling(n_games, min_periods=3).mean()
    )
    hist["h2h_run_diff"] = grp["focal_rd"].transform(
        lambda s: s.shift(1).rolling(n_games, min_periods=3).mean()
    )

    # Pull out only the home-team perspective to merge back
    home_h2h = (
        hist.rename(columns={"focal": "home_team", "opponent": "away_team",
                              "h2h_win_rate": "h2h_home_win_rate",
                              "h2h_run_diff": "h2h_home_run_diff"})
        [["Date", "home_team", "away_team", "game_id", "h2h_home_win_rate", "h2h_home_run_diff"]]
        .drop_duplicates(["game_id"])
    )

    games = games.merge(home_h2h, on=["Date", "home_team", "away_team", "game_id"], how="left")

    cov = games["h2h_home_win_rate"].notna().mean()
    print(f"  H2H coverage: {cov:.1%}  (requires 3 prior meetings)")
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
    for col in ["home_team", "away_team"]:
        if col in pl.columns:
            pl[col] = pl[col].apply(normalize_schedule_team)
    run_keys = ["game_date", "home_team", "away_team"]
    if "game_id" in pl.columns:
        run_keys.append("game_id")
    game_runs = (pl.groupby(run_keys)["runs_allowed"]
                   .sum()
                   .reset_index()
                   .rename(columns={"runs_allowed": "total_runs"}))

    # ── 2. Attach total runs to ump log ────────────────────────────────────
    ul = ump_logs.copy()
    ul["game_date"] = pd.to_datetime(ul["game_date"])
    for col in ["home_team", "away_team"]:
        if col in ul.columns:
            ul[col] = ul[col].apply(normalize_schedule_team)
    merge_keys = ["game_date", "home_team", "away_team"]
    if "game_id" in ul.columns and "game_id" in game_runs.columns:
        merge_keys.append("game_id")
    ul = ul.merge(game_runs, on=merge_keys, how="inner")
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
    # Key: (date_str, home_team, away_team, game_number) → ump_run_factor
    # Use string dates to sidestep any datetime dtype mismatches.
    ul["_date_str"] = ul["game_date"].dt.strftime("%Y-%m-%d")
    ul = ul.sort_values(["game_date", "home_team", "away_team", "game_id"]).reset_index(drop=True)
    ul["game_number"] = (
        ul.groupby(["game_date", "home_team", "away_team"], dropna=False).cumcount() + 1
    )
    ul_dedup = ul.drop_duplicates(
        subset=["_date_str", "home_team", "away_team", "game_number"], keep="first"
    )
    factor_dict: dict[tuple, float] = {
        (row["_date_str"], row["home_team"], row["away_team"], row["game_number"]): row["ump_run_factor"]
        for _, row in ul_dedup.iterrows()
        if pd.notna(row["ump_run_factor"])
    }

    games["ump_run_factor"] = [
        factor_dict.get((pd.Timestamp(d).strftime("%Y-%m-%d"), ht, at, gn), 1.0)
        for d, ht, at, gn in zip(
            games["Date"], games["home_team"], games["away_team"], games["game_number"]
        )
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
# Step 13b – Retrosheet game-state impact form
#
# These features are a lightweight, explainable version of the paper's
# game-state-delta representation.  They are built by build_state_impact_features.py
# and stored in SQLite as state_impact_features.
# ---------------------------------------------------------------------------

def attach_state_impact_features(games: pd.DataFrame,
                                 state_features: pd.DataFrame | None) -> pd.DataFrame:
    """Attach pregame rolling RE24-style form features from SQLite/CSV."""
    cols = [
        "home_offense_re24_15g", "away_offense_re24_15g", "offense_re24_diff",
        "home_sp_re24_last3", "away_sp_re24_last3", "sp_re24_diff",
        "home_bullpen_re24_15d", "away_bullpen_re24_15d", "bullpen_re24_diff",
    ]
    if state_features is None or state_features.empty:
        print("  State-impact features missing — run build_state_impact_features.py; defaulting to neutral")
        for col in cols:
            games[col] = np.nan
        return games

    sf = state_features.copy()
    sf["Date"] = pd.to_datetime(sf["game_date"])
    for col in ["home_team", "away_team"]:
        sf[col] = sf[col].apply(normalize_schedule_team)
    sf["game_number"] = pd.to_numeric(sf.get("game_number", 1), errors="coerce").fillna(1).astype(int)
    merge_keys = game_merge_keys(games, sf)
    keep = merge_keys + [c for c in cols if c in sf.columns]
    out = games.merge(sf[keep].drop_duplicates(merge_keys), on=merge_keys, how="left")
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    cov = out["home_offense_re24_15g"].notna().mean()
    print(f"  State-impact feature coverage: {cov:.1%}")
    return out


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
    # Prior-season park-adjusted SP ERA (full-season fallback)
    "home_sp_era_adj",
    "away_sp_era_adj",
    "sp_era_adj_diff",
    # In-season rolling SP ERA (last 5 starts) — higher signal
    "home_sp_inseason_era",
    "away_sp_inseason_era",
    "sp_inseason_era_diff",
    # In-season rolling SP ERA (last 3 starts) — recent form signal
    "home_sp_last3_era",
    "away_sp_last3_era",
    "sp_last3_era_diff",
    # In-season rolling bullpen RA/9 (last 15 days) — higher signal
    "home_bullpen_inseason_era",
    "away_bullpen_inseason_era",
    "bullpen_inseason_era_diff",
    # Contextual game-state impact form (Retrosheet RE24-style deltas)
    "home_offense_re24_15g",
    "away_offense_re24_15g",
    "offense_re24_diff",
    "home_sp_re24_last3",
    "away_sp_re24_last3",
    "sp_re24_diff",
    "home_bullpen_re24_15d",
    "away_bullpen_re24_15d",
    "bullpen_re24_diff",
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
    # Prior-season team platoon quality against opposing SP hand
    "home_batting_ops_vs_sp",
    "away_batting_ops_vs_sp",
    "batting_ops_vs_sp_diff",
    "home_platoon_advantage",
    "away_platoon_advantage",
    "platoon_advantage_diff",
    # Historical confirmed starting-lineup quality vs opposing SP hand
    "home_lineup_ops_vs_sp",
    "away_lineup_ops_vs_sp",
    "lineup_ops_vs_sp_diff",
    # Lineup OPS vs LHP/RHP separately (handedness split signal)
    "home_lineup_ops_vs_lhp",
    "home_lineup_ops_vs_rhp",
    "away_lineup_ops_vs_lhp",
    "away_lineup_ops_vs_rhp",
    "lineup_ops_vs_lhp_diff",
    "lineup_ops_vs_rhp_diff",
    "home_lineup_known_batters",
    "away_lineup_known_batters",
    "lineup_known_batters_diff",
    # Starting pitcher handedness context
    "home_opp_sp_is_lhp",
    "away_opp_sp_is_lhp",
    "home_sp_is_lhp",
    "away_sp_is_lhp",
    "both_sp_same_hand",
    # Prior-season Pythagorean win% (stable team quality anchor)
    "home_prior_win_pct",
    "away_prior_win_pct",
    "prior_win_pct_diff",
    # Current-season running win% + delta vs prior-year baseline
    # Captures teams that are genuinely bad/good THIS season (e.g. late-season tankers)
    "home_season_win_pct",
    "away_season_win_pct",
    "season_win_pct_diff",
    "home_season_win_pct_delta",
    "away_season_win_pct_delta",
    "season_win_pct_delta_diff",
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
    # Prior-season SP pitch stuff from FanGraphs (park-adjusted via xFIP)
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
    # SP sample size — PA batted in prior season (low = stats less reliable)
    "home_sp_pa",
    "away_sp_pa",
    "sp_pa_diff",
    # SP workload: days since last start + outs thrown in last start
    "home_sp_days_rest",
    "away_sp_days_rest",
    "sp_days_rest_diff",
    "home_sp_outs_last",
    "away_sp_outs_last",
    "sp_outs_last_diff",
    # Prior-season Statcast power metrics (barrel rate, hard hit%)
    "home_barrel_pct",
    "away_barrel_pct",
    "barrel_pct_diff",
    "home_hard_hit_pct",
    "away_hard_hit_pct",
    "hard_hit_pct_diff",
    # Home/away venue split run differential (team quality at home vs on road)
    "home_home_rd",
    "home_away_rd",
    "away_home_rd",
    "away_away_rd",
    "home_rd_split",
    "away_rd_split",
    "rd_venue_diff",
    # Rule-change era indicator (shift ban + pitch clock: 2023+)
    "post_2023_rules",
    # Calendar timing: lets the model learn early-season home/team-quality effects
    "game_month",
    "game_day_of_year",
    "is_early_season",
    # Market-aware feature. This is kept in features.csv, but training builds
    # separate market-independent and market-aware model artifacts.
    "vegas_home_prob",
]

TARGET_COL = "home_win"


def build_feature_matrix(games: pd.DataFrame) -> pd.DataFrame:
    game_dates = pd.to_datetime(games["Date"])
    # Rule-change era flag: shift ban + pitch clock took effect 2023
    games["post_2023_rules"] = (game_dates.dt.year >= 2023).astype(float)
    games["game_month"] = game_dates.dt.month.astype(float)
    games["game_day_of_year"] = game_dates.dt.dayofyear.astype(float)
    games["is_early_season"] = game_dates.dt.month.isin([3, 4]).astype(float)

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
    for col in ["home_offense_re24_15g", "away_offense_re24_15g",
                "home_sp_re24_last3", "away_sp_re24_last3",
                "home_bullpen_re24_15d", "away_bullpen_re24_15d"]:
        if col not in games.columns:
            games[col] = np.nan
    games["offense_re24_diff"] = games["home_offense_re24_15g"] - games["away_offense_re24_15g"]
    games["sp_re24_diff"] = games["home_sp_re24_last3"] - games["away_sp_re24_last3"]
    games["bullpen_re24_diff"] = games["home_bullpen_re24_15d"] - games["away_bullpen_re24_15d"]
    games["rest_diff"]                 = games["home_days_rest"]            - games["away_days_rest"]
    games["travel_diff"]               = games["away_travel_miles"]         - games["home_travel_miles"]
    games["bullpen_usage_diff"]        = games["away_bullpen_outs_3d"]      - games["home_bullpen_outs_3d"]
    games["ops_diff"]                  = games["home_team_ops"]             - games["away_team_ops"]
    games["prior_win_pct_diff"]        = games["home_prior_win_pct"]        - games["away_prior_win_pct"]
    # Current-season win% delta vs prior-year baseline (negative = underperforming)
    if "home_season_win_pct" in games.columns:
        games["home_season_win_pct_delta"] = games["home_season_win_pct"] - games["home_prior_win_pct"]
        games["away_season_win_pct_delta"] = games["away_season_win_pct"] - games["away_prior_win_pct"]
        games["season_win_pct_diff"]       = games["home_season_win_pct"] - games["away_season_win_pct"]
        games["season_win_pct_delta_diff"] = games["home_season_win_pct_delta"] - games["away_season_win_pct_delta"]
    else:
        for col in ["home_season_win_pct", "away_season_win_pct",
                    "home_season_win_pct_delta", "away_season_win_pct_delta",
                    "season_win_pct_diff", "season_win_pct_delta_diff"]:
            games[col] = np.nan
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

    # SP workload diffs
    if "home_sp_days_rest" in games.columns:
        games["sp_days_rest_diff"] = games["home_sp_days_rest"] - games["away_sp_days_rest"]
        games["sp_outs_last_diff"] = games["home_sp_outs_last"] - games["away_sp_outs_last"]
    else:
        for col in ["home_sp_days_rest", "away_sp_days_rest", "sp_days_rest_diff",
                    "home_sp_outs_last", "away_sp_outs_last", "sp_outs_last_diff"]:
            games[col] = np.nan

    # Last-3-starts SP ERA diffs (#2)
    if "home_sp_last3_era" in games.columns:
        games["sp_last3_era_diff"] = games["away_sp_last3_era"] - games["home_sp_last3_era"]
    else:
        for col in ["home_sp_last3_era", "away_sp_last3_era", "sp_last3_era_diff"]:
            games[col] = np.nan

    # Lineup LHP/RHP split diffs (#3)
    for side in ["home", "away"]:
        for hand in ["lhp", "rhp"]:
            col = f"{side}_lineup_ops_vs_{hand}"
            if col not in games.columns:
                games[col] = np.nan
    games["lineup_ops_vs_lhp_diff"] = games["home_lineup_ops_vs_lhp"] - games["away_lineup_ops_vs_lhp"]
    games["lineup_ops_vs_rhp_diff"] = games["home_lineup_ops_vs_rhp"] - games["away_lineup_ops_vs_rhp"]

    # Home/away venue split run differential (#4)
    if "home_home_rd" in games.columns:
        games["home_rd_split"]  = games["home_home_rd"] - games["home_away_rd"]
        games["away_rd_split"]  = games["away_home_rd"] - games["away_away_rd"]
        games["rd_venue_diff"]  = games["home_home_rd"] - games["away_away_rd"]
    else:
        for col in ["home_home_rd", "home_away_rd", "away_home_rd", "away_away_rd",
                    "home_rd_split", "away_rd_split", "rd_venue_diff"]:
            games[col] = np.nan

    for col in ["vegas_home_prob", "home_lineup_ops", "away_lineup_ops"]:
        if col not in games.columns:
            games[col] = np.nan
    games["lineup_ops_diff"]          = games["home_lineup_ops"]          - games["away_lineup_ops"]
    for col in ["home_lineup_ops_vs_sp", "away_lineup_ops_vs_sp",
                "home_lineup_known_batters", "away_lineup_known_batters"]:
        if col not in games.columns:
            games[col] = np.nan
    games["lineup_ops_vs_sp_diff"]    = games["home_lineup_ops_vs_sp"]    - games["away_lineup_ops_vs_sp"]
    games["lineup_known_batters_diff"] = games["home_lineup_known_batters"] - games["away_lineup_known_batters"]
    for col in ["home_batting_ops_vs_sp", "away_batting_ops_vs_sp",
                "home_platoon_advantage", "away_platoon_advantage",
                "home_opp_sp_is_lhp", "away_opp_sp_is_lhp",
                "home_sp_is_lhp", "away_sp_is_lhp", "both_sp_same_hand"]:
        if col not in games.columns:
            games[col] = np.nan
    games["batting_ops_vs_sp_diff"]   = games["home_batting_ops_vs_sp"]   - games["away_batting_ops_vs_sp"]
    games["platoon_advantage_diff"]   = games["home_platoon_advantage"]   - games["away_platoon_advantage"]

    available = [c for c in FEATURE_COLS if c in games.columns]
    missing = [c for c in FEATURE_COLS if c not in games.columns]
    coverage_pct = len(available) / len(FEATURE_COLS) * 100
    if missing:
        print(f"  WARNING: {len(missing)}/{len(FEATURE_COLS)} feature cols absent "
              f"({coverage_pct:.0f}% coverage): {missing[:10]}{'…' if len(missing) > 10 else ''}")

    for col in ["game_id", "game_number", "game_pk"]:
        if col not in games.columns:
            games[col] = np.nan
    id_cols = [
        "game_id", "game_number", "game_pk",
        "Date", "year", "home_team", "away_team",
        "home_runs", "away_runs", TARGET_COL,
    ]
    out = games[id_cols + available].copy()
    out = apply_feature_defaults(out, available)
    # Drop rows where either rolling_rd is NaN — these are early-season games
    # with < min_periods prior games and would bias training with noisy inputs.
    out = out.dropna(subset=["home_rolling_rd", "away_rolling_rd"], how="any")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import database as _db

    _db_path = _db.DB_PATH
    if not os.path.exists(_db_path):
        raise FileNotFoundError(
            f"{_db_path} not found — run python migrate_to_db.py first."
        )

    batting_path = os.path.join(DATA_DIR, "batting_stats.csv")
    if not _db.table_exists("batting_stats") and not os.path.exists(batting_path):
        raise FileNotFoundError(
            "No batting_stats table or data/batting_stats.csv found — run data_ingestion.py first."
        )

    print("Loading data from mlb.db...")
    _conn = _db.get_connection()

    # --- Game results (unified historical + live, already cleaned) ---
    _gl = pd.read_sql("SELECT * FROM game_logs ORDER BY game_date, home_team, game_number", _conn)
    _gl["Date"] = pd.to_datetime(_gl["game_date"])
    _gl = _gl.rename(columns={"game_date": "_game_date_str"})
    games = _gl.copy()
    for col in ["home_team", "away_team"]:
        games[col] = games[col].apply(normalize_schedule_team)
    print(f"  {len(games):,} games from game_logs table")

    # --- Pitcher game logs (unified historical + live) ---
    game_logs = pd.read_sql("SELECT * FROM pitcher_game_logs", _conn)
    game_logs["game_date"] = pd.to_datetime(game_logs["game_date"])
    for col in ["home_team", "away_team"]:
        if col in game_logs.columns:
            game_logs[col] = game_logs[col].apply(normalize_schedule_team)
    print(f"  {len(game_logs):,} rows from pitcher_game_logs table")

    # --- SP assignments (unified historical + live) ---
    _gs = pd.read_sql("SELECT * FROM game_starters", _conn)
    _gs["Date"] = pd.to_datetime(_gs["game_date"])
    game_sp = _gs.copy()
    for col in ["home_team", "away_team"]:
        game_sp[col] = game_sp[col].apply(normalize_schedule_team)
    print(f"  {len(game_sp):,} rows from game_starters table")

    # --- Umpire logs ---
    _ump_raw = pd.read_sql("SELECT * FROM umpire_game_logs", _conn)
    ump_logs = _ump_raw if not _ump_raw.empty else None
    if ump_logs is not None:
        ump_logs["game_date"] = pd.to_datetime(ump_logs["game_date"])
        for col in ["home_team", "away_team"]:
            if col in ump_logs.columns:
                ump_logs[col] = ump_logs[col].apply(normalize_schedule_team)

    # --- IL counts ---
    _il_raw = pd.read_sql("SELECT * FROM il_counts", _conn)
    il_df = _il_raw if not _il_raw.empty else None
    if il_df is not None:
        il_df["date"] = pd.to_datetime(il_df["date"])

    # --- Static per-season data ---
    pitcher_stats = _db.read_table_or_csv(
        "pitcher_stats", os.path.join(DATA_DIR, "pitcher_stats.csv"), conn=_conn
    )
    batting_stats = _db.read_table_or_csv(
        "batting_stats", batting_path, conn=_conn
    )

    _conn.close()
    print(f"  {len(games):,} games total (DB)")

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
    _conn2 = _db.get_connection()
    _stuff_raw = pd.read_sql("SELECT * FROM pitcher_stuff", _conn2)
    if not _stuff_raw.empty:
        pitcher_stuff = _stuff_raw.rename(columns={
            "throws": "Throws", "fbv": "FBv", "swstr_pct": "SwStr_pct",
            "k_pct": "K_pct", "xfip": "xFIP", "gs": "GS",
        })
        games = attach_sp_stuff(games, pitcher_stuff)
    else:
        print("  pitcher_stuff table empty — run fetch_pitcher_stuff.py first")
        games = attach_sp_stuff(games, None)

    print("Computing rest days and travel distance...")
    games = compute_rest_and_travel(games)

    print("Attaching prior-season Statcast batting (barrel rate, hard hit%)...")
    statcast_df = pd.read_sql("SELECT * FROM statcast_batting", _conn2)
    games = attach_statcast_batting(games, statcast_df if not statcast_df.empty else None)

    print("Attaching prior-season batting quality (team OPS)...")
    games = attach_batting_quality(games, batting_stats)

    print("Attaching prior-season batting splits vs opposing SP hand...")
    splits_df = pd.read_sql("SELECT * FROM team_splits", _conn2)
    if not splits_df.empty:
        splits_df = splits_df.drop_duplicates(subset=["team", "year"], keep="last")
        splits_df["year"] = splits_df["year"].astype(int)
    games = attach_batting_splits_vs_sp(games, splits_df if not splits_df.empty else None)

    print("Attaching historical starting-lineup split OPS...")
    _lineups_raw = pd.read_sql("SELECT * FROM historical_lineups", _conn2)
    games = attach_historical_lineups(games, _lineups_raw if not _lineups_raw.empty else None)

    print("Attaching prior-season Pythagorean win%...")
    games = attach_prior_win_pct(games)

    print("Computing head-to-head history...")
    games = compute_h2h_stats(games)

    print("Attaching weather data...")
    _weather_raw = pd.read_sql("SELECT * FROM game_weather", _conn2)
    _weather_raw = _weather_raw.rename(columns={"game_date": "date"})
    _weather_raw["date"] = pd.to_datetime(_weather_raw["date"])
    games = attach_weather(games, _weather_raw)

    print("Attaching umpire run factor...")
    if ump_logs is not None:
        games = attach_umpire_factor(games, ump_logs, game_logs)
    else:
        print("  umpire_game_logs table empty — run parse_retrosheet_events.py; defaulting to 1.0")
        games["ump_run_factor"] = 1.0

    print("Attaching IL (Injured List) counts...")
    if il_df is not None:
        games = attach_il_counts(games, il_df)
    else:
        print("  il_counts table empty — run fetch_il_data.py; defaulting to 0")
        games["home_il_count"] = 0
        games["away_il_count"] = 0
        games["il_diff"]       = 0

    print("Attaching IL quality (WAR-weighted)...")
    il_quality_path = os.path.join(DATA_DIR, "il_quality.csv")
    if _db.table_exists("il_quality", _conn2) or os.path.exists(il_quality_path):
        il_quality_df = _db.read_table_or_csv(
            "il_quality", il_quality_path, parse_dates=["date"], conn=_conn2
        )

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

    print("Attaching Retrosheet state-impact form features...")
    state_impact_path = os.path.join(DATA_DIR, "state_impact_features.csv")
    if _db.table_exists("state_impact_features", _conn2) or os.path.exists(state_impact_path):
        state_impact_df = _db.read_table_or_csv(
            "state_impact_features", state_impact_path, parse_dates=["game_date"], conn=_conn2
        )
    else:
        state_impact_df = None
    games = attach_state_impact_features(games, state_impact_df)

    print("Attaching historical Vegas consensus odds...")
    _odds_raw = pd.read_sql(
        "SELECT game_date, home_team, away_team, game_number, vegas_home_prob "
        "FROM game_odds WHERE vegas_home_prob IS NOT NULL",
        _conn2,
    )
    _conn2.close()
    if not _odds_raw.empty:
        _odds_raw["Date"] = pd.to_datetime(_odds_raw["game_date"])
        _odds_raw = _odds_raw.drop(columns=["game_date"])
        for col in ["home_team", "away_team"]:
            _odds_raw[col] = _odds_raw[col].apply(normalize_schedule_team)
        _odds_raw = _odds_raw.drop_duplicates(["Date", "home_team", "away_team", "game_number"])
        _merge_keys = game_merge_keys(games, _odds_raw)
        games = games.merge(
            _odds_raw[_merge_keys + ["vegas_home_prob"]],
            on=_merge_keys, how="left"
        )
        coverage = games["vegas_home_prob"].notna().mean()
        n_games  = games["vegas_home_prob"].notna().sum()
        print(f"  Vegas odds coverage: {coverage:.1%} of games ({n_games:,} games)")
    else:
        print("  game_odds table empty — run fetch_historical_odds.py first")
        games["vegas_home_prob"] = np.nan

    print("Building feature matrix...")
    features = build_feature_matrix(games)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Columns: {features.columns.tolist()}")
    print(f"  Home win rate: {features[TARGET_COL].mean():.3f}")

    out_path = os.path.join(DATA_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    _conn3 = _db.get_connection()
    _db.replace_table(features, "features", _conn3)
    _conn3.close()
    print(f"\nSaved to {out_path} — run train_model.py next.")
