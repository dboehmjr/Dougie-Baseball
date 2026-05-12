"""
Generate pre-game win probabilities for a new matchup.

Usage:
    python predict.py --home NYY --away BOS --year 2025

The script loads the saved model and looks up rolling stats for both teams
from the feature matrix.  For truly live predictions you would replace the
feature lookup with a real-time data pull.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import joblib
import pandas as pd
import numpy as np

from fetch_weather import get_game_weather, wind_to_cf, FULL_DOME
from fetch_il_data import get_current_il_count, get_current_il_quality
from feature_engineering import PARK_DIMENSIONS
from fetch_pitcher_stuff import get_pitcher_stuff, fetch_pitcher_stuff, _normalize_name
from fetch_statcast_batting import get_team_statcast
from fetch_odds import load_or_fetch_odds, get_home_implied_prob, get_moneyline_str
from fetch_splits import fetch_all_splits, get_team_split_ops
from fetch_lineups import fetch_confirmed_lineups, get_lineup_ops
from feature_defaults import apply_feature_defaults

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------------------
# Module-level data cache — keyed by (path, mtime) so it auto-refreshes
# when a file changes on disk, but doesn't re-read on every prediction call.
# ---------------------------------------------------------------------------
_FILE_CACHE: dict = {}
_MODEL_CACHE: dict = {}
_DB_CACHE: dict = {}


def _load_csv(path: str, **kwargs) -> pd.DataFrame | None:
    """Read a CSV once per file-modification-time; return cached copy thereafter."""
    if not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    if key not in _FILE_CACHE:
        _FILE_CACHE[key] = pd.read_csv(path, **kwargs)
    return _FILE_CACHE[key]


def _load_db_table(table: str,
                   parse_dates: list[str] | None = None,
                   rename: dict | None = None,
                   query: str | None = None) -> pd.DataFrame | None:
    """Read a SQLite table from mlb.db, falling back to CSV callers on failure."""
    try:
        import database as db_mod
        db_path = db_mod.DB_PATH
        if not os.path.exists(db_path):
            return None
        wal_path = f"{db_path}-wal"
        db_mtime = max(
            os.path.getmtime(db_path),
            os.path.getmtime(wal_path) if os.path.exists(wal_path) else 0,
        )
        key = (table, query or table, db_mtime)
        if key not in _DB_CACHE:
            conn = db_mod.get_connection()
            sql = query or f'SELECT * FROM "{table}"'
            df = pd.read_sql(sql, conn)
            conn.close()
            if rename:
                df = df.rename(columns=rename)
            for col in parse_dates or []:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            _DB_CACHE[key] = df
        return _DB_CACHE[key].copy()
    except Exception:
        return None


def _load_prediction_game_logs() -> pd.DataFrame | None:
    df = _load_db_table("game_logs", parse_dates=["Date"],
                        rename={"game_date": "Date"})
    if df is not None and not df.empty:
        return df
    return _load_csv(os.path.join(DATA_DIR, "game_logs_live.csv"), parse_dates=["Date"])


def _load_pitcher_game_logs() -> pd.DataFrame | None:
    df = _load_db_table("pitcher_game_logs", parse_dates=["game_date"])
    if df is not None and not df.empty:
        return df
    gl_path = os.path.join(DATA_DIR, "pitcher_game_logs.csv")
    gl_live_path = os.path.join(DATA_DIR, "pitcher_game_logs_live.csv")
    game_logs_df = _load_csv(gl_path, parse_dates=["game_date"])
    live_gl = (_load_csv(gl_live_path, parse_dates=["game_date"])
               if os.path.exists(gl_live_path) else None)
    if live_gl is not None and not live_gl.empty:
        if game_logs_df is not None and not game_logs_df.empty:
            return pd.concat([game_logs_df, live_gl], ignore_index=True).drop_duplicates(
                subset=["game_id", "pitcher_id"]
            )
        return live_gl
    return game_logs_df


def _load_weather_data() -> pd.DataFrame | None:
    df = _load_db_table("game_weather", parse_dates=["date"],
                        rename={"game_date": "date"})
    if df is not None and not df.empty:
        return df
    return _load_csv(os.path.join(DATA_DIR, "weather.csv"), parse_dates=["date"])


def _load_pitcher_stuff_data() -> pd.DataFrame | None:
    df = _load_db_table("pitcher_stuff")
    if df is not None and not df.empty:
        return df.rename(columns={
            "throws": "Throws", "fbv": "FBv", "swstr_pct": "SwStr_pct",
            "k_pct": "K_pct", "xfip": "xFIP", "gs": "GS",
        })
    return _load_csv(os.path.join(DATA_DIR, "pitcher_stuff.csv"))


def _load_table_or_csv(table: str, csv_name: str,
                       parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    df = _load_db_table(table, parse_dates=parse_dates)
    if df is not None and not df.empty:
        return df
    return _load_csv(os.path.join(DATA_DIR, csv_name), parse_dates=parse_dates)


def _load_model(model_path: str) -> dict:
    """Load the model artifact once; reload only if the file changes."""
    key = (model_path, os.path.getmtime(model_path))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()   # only keep the most recent version
        _MODEL_CACHE[key] = joblib.load(model_path)
    return _MODEL_CACHE[key]


_IL_LIVE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "il_live_cache.json")
_LIVE_IL_DAYS = 14   # use live API for games within this many days of today


def _load_il_live_cache() -> dict:
    try:
        if os.path.exists(_IL_LIVE_CACHE_PATH):
            with open(_IL_LIVE_CACHE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_il_live_cache(cache: dict) -> None:
    try:
        with open(_IL_LIVE_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _il_from_snapshot(team: str, game_date: pd.Timestamp,
                       il_df: pd.DataFrame | None) -> float:
    """
    Return IL count for a team on a given date.

    For games within _LIVE_IL_DAYS of today: hits the live MLB Stats API and
    caches results in il_live_cache.json so each team is only fetched once per day.

    For historical games: uses the monthly snapshot CSV (fast, no API call).
    """
    today = pd.Timestamp.today().normalize()
    days_ago = (today - game_date).days

    if days_ago <= _LIVE_IL_DAYS:
        date_str = str(game_date.date())
        cache = _load_il_live_cache()
        if date_str in cache and team in cache[date_str]:
            return float(cache[date_str][team])
        # Fetch live
        try:
            count = get_current_il_count(team, date_str)
            if not np.isnan(count):
                cache.setdefault(date_str, {})[team] = int(count)
                _save_il_live_cache(cache)
                return float(count)
        except Exception:
            pass

    # Fall back to monthly snapshot
    if il_df is None or il_df.empty:
        return np.nan
    t_df = il_df[il_df["team"] == team].sort_values("date")
    past = t_df[t_df["date"] <= game_date]
    if past.empty:
        return np.nan
    return float(past.iloc[-1]["il_count"])

PARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4453, -112.0667), "ATL": (33.8908,  -84.4677),
    "BAL": (39.2838,  -76.6218), "BOS": (42.3467,  -71.0972),
    "CHC": (41.9484,  -87.6553), "CHW": (41.8300,  -87.6338),
    "CIN": (39.0979,  -84.5082), "CLE": (41.4962,  -81.6852),
    "COL": (39.7559, -104.9942), "DET": (42.3390,  -83.0485),
    "HOU": (29.7573,  -95.3555), "KCR": (39.0517,  -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781,  -80.2197), "MIL": (43.0280,  -87.9712),
    "MIN": (44.9817,  -93.2781), "NYM": (40.7571,  -73.8458),
    "NYY": (40.8296,  -73.9262), "ATH": (37.7516, -122.2005),
    "PHI": (39.9061,  -75.1665), "PIT": (40.4469,  -80.0057),
    "SDP": (32.7076, -117.1570), "SEA": (47.5914, -122.3325),
    "SFG": (37.7786, -122.3893), "STL": (38.6226,  -90.1928),
    "TBR": (27.7683,  -82.6534), "TEX": (32.7473,  -97.0831),
    "TOR": (43.6414,  -79.3894), "WSN": (38.8730,  -77.0074),
}


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def compute_rest_travel(team: str,
                         game_date: pd.Timestamp,
                         current_park: str,
                         features_df: pd.DataFrame) -> dict:
    """
    Given a team and the date of the upcoming game, find their most recent
    previous game and return days_rest and travel_miles.
    """
    mask = (
        ((features_df["home_team"] == team) | (features_df["away_team"] == team))
        & (features_df["Date"] < game_date)
    )
    past = features_df[mask].sort_values("Date")
    if past.empty:
        return {"days_rest": np.nan, "travel_miles": np.nan}

    last         = past.iloc[-1]
    prev_date    = pd.to_datetime(last["Date"])
    prev_park    = last["home_team"]   # game location is always home team's park

    days_rest = max(0, min(int((game_date - prev_date).days) - 1, 7))

    if prev_park == current_park:
        travel_miles = 0.0
    else:
        c1 = PARK_COORDS.get(prev_park)
        c2 = PARK_COORDS.get(current_park)
        travel_miles = _haversine_miles(*c1, *c2) if (c1 and c2) else np.nan

    return {"days_rest": days_rest, "travel_miles": travel_miles}


TEAM_ABBREV_HELP = (
    "ARI ATL BAL BOS CHC CHW CIN CLE COL DET "
    "HOU KCR LAA LAD MIA MIL MIN NYM NYY ATH "
    "PHI PIT SDP SEA SFG STL TBR TEX TOR WSN"
)

TEAM_ALIASES = {
    "OAK": "ATH",
    "ATH": "ATH",
}


def canonical_team(team: str) -> str:
    abbr = str(team or "").strip().upper()
    return TEAM_ALIASES.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Feature lookup
# ---------------------------------------------------------------------------

def get_team_ops(team: str, year: int, batting_stats_df: pd.DataFrame) -> float:
    """
    Return team's PA-weighted OPS for the given year from cached batting stats.
    Used to look up prior-year OPS at prediction time.
    """
    df = batting_stats_df.copy()
    df = df[
        (df["year"] == year)
        & df["Lev"].str.startswith("Maj", na=False)
        & ~df["Tm"].str.contains(",", na=False)
    ].copy()
    df["PA"]  = pd.to_numeric(df["PA"],  errors="coerce").fillna(0)
    df["OPS"] = pd.to_numeric(df["OPS"], errors="coerce")
    df = df[df["PA"] >= 10].dropna(subset=["OPS"])

    # Normalize team names (same logic as feature_engineering)
    def _norm(tm, lev):
        tm_l, lev_l = str(tm).strip().lower(), str(lev).strip().lower()
        if tm_l == "chicago":
            tm_l = "chicago white sox" if "al" in lev_l else "chicago cubs"
        elif tm_l == "new york":
            tm_l = "new york yankees" if "al" in lev_l else "new york mets"
        elif tm_l == "los angeles":
            tm_l = "los angeles angels" if "al" in lev_l else "los angeles dodgers"
        ABBREV = {
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
            "oakland athletics": "ATH", "philadelphia phillies": "PHI",
            "pittsburgh pirates": "PIT", "san diego padres": "SDP",
            "seattle mariners": "SEA", "san francisco giants": "SFG",
            "st. louis cardinals": "STL", "tampa bay rays": "TBR",
            "texas rangers": "TEX", "toronto blue jays": "TOR",
            "washington nationals": "WSN",
            "arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL",
            "boston": "BOS", "cincinnati": "CIN", "cleveland": "CLE",
            "colorado": "COL", "detroit": "DET", "houston": "HOU",
            "kansas city": "KCR", "miami": "MIA", "milwaukee": "MIL",
            "minnesota": "MIN", "oakland": "ATH", "philadelphia": "PHI",
            "pittsburgh": "PIT", "san diego": "SDP", "seattle": "SEA",
            "san francisco": "SFG", "st. louis": "STL", "tampa bay": "TBR",
            "texas": "TEX", "toronto": "TOR", "washington": "WSN",
        }
        return ABBREV.get(tm_l, tm_l.upper())

    df["team_abbr"] = df.apply(lambda r: _norm(r["Tm"], r["Lev"]), axis=1)
    subset = df[df["team_abbr"] == team]
    if subset.empty:
        return np.nan
    return float(np.average(subset["OPS"], weights=subset["PA"].clip(lower=1)))


def compute_sp_workload_live(sp_name_norm: str | None,
                              sp_id: str | None,
                              game_date: pd.Timestamp,
                              game_logs_df: pd.DataFrame) -> dict:
    """
    Return days_rest and outs_last_start for a starting pitcher going into game_date.
    Matches by pitcher_id first, falls back to normalized name match.
    """
    empty = {"sp_days_rest": np.nan, "sp_outs_last": np.nan}
    if game_logs_df is None or game_logs_df.empty:
        return empty

    gl = game_logs_df[game_logs_df["is_starter"]].copy()
    gl["game_date"] = pd.to_datetime(gl["game_date"])

    # Match on pitcher_id if available, else on normalized name
    if sp_id:
        pitcher_starts = gl[gl["pitcher_id"] == sp_id]
    elif sp_name_norm:
        from fetch_pitcher_stuff import _normalize_name
        gl["name_norm"] = gl["pitcher_name"].apply(
            lambda n: _normalize_name(str(n)) if pd.notna(n) else ""
        )
        pitcher_starts = gl[gl["name_norm"] == sp_name_norm]
    else:
        return empty

    prior = pitcher_starts[pitcher_starts["game_date"] < game_date].sort_values("game_date")
    if prior.empty:
        return empty

    last_start = prior.iloc[-1]
    days_rest  = int(np.clip((game_date - last_start["game_date"]).days - 1, 0, 10))
    outs_last  = float(last_start["outs_recorded"])
    return {"sp_days_rest": days_rest, "sp_outs_last": outs_last}


def compute_bullpen_usage_live(team: str,
                               game_date: pd.Timestamp,
                               game_logs_df: pd.DataFrame,
                               window_days: int = 3) -> float:
    """
    Sum outs recorded by a team's relievers in the `window_days` calendar
    days immediately before game_date (exclusive of game_date itself).
    """
    cutoff = game_date - pd.Timedelta(days=window_days)
    mask = (
        (~game_logs_df["is_starter"])
        & (game_logs_df["game_date"] >= cutoff)
        & (game_logs_df["game_date"] < game_date)
        & (
            ((game_logs_df["home_team"] == team) & (game_logs_df["team_side"] == 1))
            | ((game_logs_df["away_team"] == team) & (game_logs_df["team_side"] == 0))
        )
    )
    return float(game_logs_df[mask]["outs_recorded"].sum())


def compute_h2h_live(home_team: str,
                     away_team: str,
                     game_date: pd.Timestamp,
                     features_df: pd.DataFrame,
                     n_games: int = 10,
                     min_games: int = 3) -> dict:
    """
    Compute rolling head-to-head stats for home_team vs away_team
    using only current-season prior meetings recorded in features_df.
    Returns home team's win rate and avg run differential.
    """
    current_season = game_date.year
    # Find all prior games between these two teams this season only
    mask = (
        (
            ((features_df["home_team"] == home_team) & (features_df["away_team"] == away_team)) |
            ((features_df["home_team"] == away_team) & (features_df["away_team"] == home_team))
        )
        & (features_df["Date"] < game_date)
        & (features_df["Date"].dt.year == current_season)
    )
    prior = features_df[mask].sort_values("Date").tail(n_games)

    if len(prior) < min_games:
        return {"h2h_home_win_rate": np.nan, "h2h_home_run_diff": np.nan}

    wins, rds = [], []
    for _, row in prior.iterrows():
        if row["home_team"] == home_team:
            # home_team was home in this prior game
            wins.append(row["home_win"])
            rds.append(row.get("home_runs", np.nan) - row.get("away_runs", np.nan))
        else:
            # home_team was the away team in this prior game
            wins.append(1 - row["home_win"])
            rds.append(row.get("away_runs", np.nan) - row.get("home_runs", np.nan))

    return {
        "h2h_home_win_rate": float(np.nanmean(wins)),
        "h2h_home_run_diff": float(np.nanmean(rds)),
    }


def _compute_season_win_pct(team: str, year: int,
                             features_df: pd.DataFrame,
                             live_logs: pd.DataFrame | None,
                             min_games: int = 10) -> float:
    """
    Compute season win% from scratch for the given team/year using all games
    in features_df plus any newer games in live_logs.  Returns NaN until
    min_games have been played (matches the training feature's min_periods=10).
    """
    mask = (
        ((features_df["home_team"] == team) | (features_df["away_team"] == team))
        & (features_df["year"] == year)
        & (features_df["home_win"].notna())
    )
    season_games = features_df[mask].sort_values("Date")

    wins = total = 0
    for _, row in season_games.iterrows():
        is_home = row["home_team"] == team
        wins  += int((row["home_win"] == 1) if is_home else (row["home_win"] == 0))
        total += 1

    if live_logs is not None and not live_logs.empty:
        last_date = season_games["Date"].max() if not season_games.empty else pd.Timestamp("2000-01-01")
        ll = live_logs.copy()
        ll["Date"] = pd.to_datetime(ll["Date"])
        newer = ll[
            (ll["Date"] > last_date)
            & (ll["Date"].dt.year == year)
            & ((ll["home_team"] == team) | (ll["away_team"] == team))
            & (ll["home_win"].notna())
        ]
        for _, row in newer.iterrows():
            is_home = row["home_team"] == team
            wins  += int((row["home_win"] == 1) if is_home else (row["home_win"] == 0))
            total += 1

    return wins / total if total >= min_games else np.nan


def _compute_current_streak(team: str, last_features_date: pd.Timestamp,
                             last_streak: float, live_logs: pd.DataFrame | None) -> float:
    """
    Update a streak by replaying any game results in live_logs that occurred
    after the last features.csv snapshot date.
    Streak sign convention: positive = win streak length, negative = loss streak length.
    """
    if live_logs is None or live_logs.empty:
        return last_streak

    live_logs = live_logs.copy()
    live_logs["Date"] = pd.to_datetime(live_logs["Date"])
    newer = live_logs[live_logs["Date"] >= last_features_date].sort_values("Date")
    team_games = newer[(newer["home_team"] == team) | (newer["away_team"] == team)]

    streak = last_streak
    for _, row in team_games.iterrows():
        is_home = row["home_team"] == team
        won = (row["home_win"] == 1) if is_home else (row["home_win"] == 0)
        if won:
            streak = (streak + 1) if streak > 0 else 1
        else:
            streak = (streak - 1) if streak < 0 else -1
    return streak


def get_latest_team_features(team: str,
                              year: int,
                              features_df: pd.DataFrame,
                              live_logs: pd.DataFrame | None = None) -> dict:
    """Return the most recent feature snapshot for a team as a pre-game proxy."""
    mask = (
        ((features_df["home_team"] == team) | (features_df["away_team"] == team))
        & (features_df["year"] == year)
    )
    subset = features_df[mask].sort_values("Date")

    if subset.empty:
        mask2 = (features_df["home_team"] == team) | (features_df["away_team"] == team)
        subset = features_df[mask2].sort_values("Date")

    if subset.empty:
        raise ValueError(f"No historical data found for team '{team}'")

    last = subset.iloc[-1]

    is_home = last["home_team"] == team
    prefix = "home" if is_home else "away"

    raw_streak = float(last.get(f"{prefix}_streak", 0.0))
    current_streak = _compute_current_streak(team, pd.Timestamp(last["Date"]),
                                             raw_streak, live_logs)

    prior_win_pct   = last.get(f"{prefix}_prior_win_pct", np.nan)
    season_win_pct  = _compute_season_win_pct(team, year, features_df, live_logs)

    return {
        "rolling_rd":           last.get(f"{prefix}_rolling_rd",           np.nan),
        "rolling_rs":           last.get(f"{prefix}_rolling_rs",           np.nan),
        "sp_era":               last.get(f"{prefix}_sp_era_adj",           np.nan),
        "sp_inseason_era":      last.get(f"{prefix}_sp_inseason_era",      np.nan),
        "bullpen_inseason_era": last.get(f"{prefix}_bullpen_inseason_era", np.nan),
        "park_factor":          last.get("park_factor", np.nan) if is_home else np.nan,
        "last7_rd":             last.get(f"{prefix}_last7_rd",             np.nan),
        "streak":               current_streak,
        "momentum":             last.get(f"{prefix}_momentum",             np.nan),
        "prior_win_pct":        prior_win_pct,
        "season_win_pct":       season_win_pct,
        "season_win_pct_delta": (season_win_pct - prior_win_pct
                                  if not np.isnan(season_win_pct) and not np.isnan(prior_win_pct)
                                  else np.nan),
    }


def get_ump_run_factor(ump_name: str,
                       game_date: pd.Timestamp,
                       pitcher_logs: pd.DataFrame,
                       ump_logs: pd.DataFrame) -> float:
    """
    Look up an umpire's historical run factor (vs. league average) using
    only games strictly before game_date.  Returns 1.0 if unknown / < 20 games.
    """
    if ump_name is None or ump_logs is None or ump_logs.empty:
        return 1.0

    ump_logs = ump_logs.copy()
    ump_logs["game_date"] = pd.to_datetime(ump_logs["game_date"])
    pitcher_logs = pitcher_logs.copy()
    pitcher_logs["game_date"] = pd.to_datetime(pitcher_logs["game_date"])

    # Total runs per game
    game_runs = (pitcher_logs.groupby(["game_date", "home_team"])["runs_allowed"]
                              .sum().reset_index()
                              .rename(columns={"runs_allowed": "total_runs"}))
    ul = ump_logs.merge(game_runs, on=["game_date", "home_team"], how="inner")
    ul["year"] = ul["game_date"].dt.year

    # League avg runs/game by year (using all historical data for simplicity)
    league_avg = ul.groupby("year")["total_runs"].mean().to_dict()
    year_key   = game_date.year
    lg_avg     = league_avg.get(year_key, 9.0)

    # Ump's prior games (before game_date)
    prior = ul[(ul["ump_name"] == ump_name) & (ul["game_date"] < game_date)]
    if len(prior) < 20:
        return 1.0
    return float(prior["total_runs"].mean() / lg_avg)


def _safe_diff(a, b):
    """Return a - b, or NaN if either value is missing/unknown."""
    a_missing = a is None or (a != a)  # None or NaN
    b_missing = b is None or (b != b)
    if a_missing or b_missing:
        return np.nan
    return float(a) - float(b)


def _get_recent_sp_name(team: str, game_date: pd.Timestamp,
                         features_df: pd.DataFrame, side: str) -> str | None:
    """Return the most recent normalized SP name for a team from features history."""
    col = f"{side}_sp_norm" if f"{side}_sp_norm" in features_df.columns else None
    if col is None:
        return None
    team_col = f"{side}_team"
    mask = (features_df[team_col] == team) & (features_df["Date"] < game_date)
    rows = features_df[mask].sort_values("Date")
    if rows.empty or col not in rows.columns:
        return None
    recent = rows[col].dropna()
    return recent.iloc[-1] if not recent.empty else None


def build_input_row(home: dict, away: dict) -> pd.DataFrame:
    park_dims = PARK_DIMENSIONS.get(home.get("team", ""), {})
    return pd.DataFrame([{
        "game_month":                home.get("game_month", np.nan),
        "game_day_of_year":          home.get("game_day_of_year", np.nan),
        "is_early_season":           home.get("is_early_season", np.nan),
        "home_rolling_rd":           home["rolling_rd"],
        "away_rolling_rd":           away["rolling_rd"],
        "rd_diff":                   home["rolling_rd"]           - away["rolling_rd"],
        "home_rolling_rs":           home["rolling_rs"],
        "away_rolling_rs":           away["rolling_rs"],
        "rs_diff":                   home["rolling_rs"]           - away["rolling_rs"],
        "home_sp_era_adj":           home["sp_era"],
        "away_sp_era_adj":           away["sp_era"],
        "sp_era_adj_diff":           away["sp_era"]               - home["sp_era"],
        "home_sp_inseason_era":      home["sp_inseason_era"],
        "away_sp_inseason_era":      away["sp_inseason_era"],
        "sp_inseason_era_diff":      away["sp_inseason_era"]      - home["sp_inseason_era"],
        "home_bullpen_inseason_era": home["bullpen_inseason_era"],
        "away_bullpen_inseason_era": away["bullpen_inseason_era"],
        "bullpen_inseason_era_diff": away["bullpen_inseason_era"] - home["bullpen_inseason_era"],
        "park_factor":               home["park_factor"],
        # Rest & travel
        "home_days_rest":    home.get("days_rest",    np.nan),
        "away_days_rest":    away.get("days_rest",    np.nan),
        "rest_diff":         _safe_diff(home.get("days_rest"),    away.get("days_rest")),
        "away_travel_miles": away.get("travel_miles", np.nan),
        "travel_diff":       _safe_diff(away.get("travel_miles"), home.get("travel_miles")),
        # Head-to-head history
        "h2h_home_win_rate": home.get("h2h_home_win_rate", np.nan),
        "h2h_home_run_diff": home.get("h2h_home_run_diff", np.nan),
        # Bullpen usage / fatigue
        "home_bullpen_outs_3d": home.get("bullpen_outs_3d", np.nan),
        "away_bullpen_outs_3d": away.get("bullpen_outs_3d", np.nan),
        "bullpen_usage_diff":   _safe_diff(away.get("bullpen_outs_3d"), home.get("bullpen_outs_3d")),
        # Short-term form (7-game rolling)
        "home_last7_rd":   home.get("last7_rd",   np.nan),
        "away_last7_rd":   away.get("last7_rd",   np.nan),
        "last7_rd_diff":   _safe_diff(home.get("last7_rd"),  away.get("last7_rd")),
        # Momentum (heating up vs cooling down)
        "home_momentum":   home.get("momentum",   np.nan),
        "away_momentum":   away.get("momentum",   np.nan),
        "momentum_diff":   _safe_diff(home.get("momentum"),  away.get("momentum")),
        # Win/loss streak
        "home_streak":     home.get("streak", 0),
        "away_streak":     away.get("streak", 0),
        "streak_diff":     _safe_diff(home.get("streak"), away.get("streak")),
        # Prior-season team batting OPS
        "home_team_ops": home.get("team_ops", np.nan),
        "away_team_ops": away.get("team_ops", np.nan),
        "ops_diff":      _safe_diff(home.get("team_ops"), away.get("team_ops")),
        # Game-time weather
        "temp_f":         home.get("temp_f",         np.nan),
        "wind_speed_mph": home.get("wind_speed_mph",  np.nan),
        "wind_to_cf":     home.get("wind_to_cf",      np.nan),
        "humidity_pct":   home.get("humidity_pct",    np.nan),
        # Umpire tendency
        "ump_run_factor": home.get("ump_run_factor", 1.0),
        # Injured List counts
        "home_il_count": home.get("il_count", 0),
        "away_il_count": away.get("il_count", 0),
        "il_diff":       _safe_diff(home.get("il_count"), away.get("il_count")),
        # IL quality: prior-year WAR of players on the IL
        "home_il_war":   home.get("il_war", np.nan),
        "away_il_war":   away.get("il_war", np.nan),
        "il_war_diff":   _safe_diff(home.get("il_war"), away.get("il_war")),
        # Park dimensions (static per home park)
        "park_lf_dist":    park_dims.get("lf_dist",   np.nan),
        "park_cf_dist":    park_dims.get("cf_dist",   np.nan),
        "park_rf_dist":    park_dims.get("rf_dist",   np.nan),
        "park_lf_wall_ht": park_dims.get("lf_wall_ht", np.nan),
        "park_altitude_ft": park_dims.get("altitude_ft", np.nan),
        # SP pitch stuff (FanGraphs season-level)
        "home_sp_fbv":   home.get("sp_fbv",   np.nan),
        "away_sp_fbv":   away.get("sp_fbv",   np.nan),
        "sp_fbv_diff":   _safe_diff(home.get("sp_fbv"),   away.get("sp_fbv")),
        "home_sp_swstr": home.get("sp_swstr", np.nan),
        "away_sp_swstr": away.get("sp_swstr", np.nan),
        "sp_swstr_diff": _safe_diff(home.get("sp_swstr"), away.get("sp_swstr")),
        "home_sp_k_pct": home.get("sp_k_pct", np.nan),
        "away_sp_k_pct": away.get("sp_k_pct", np.nan),
        "sp_k_pct_diff": _safe_diff(home.get("sp_k_pct"), away.get("sp_k_pct")),
        "home_sp_xfip":  home.get("sp_xfip",  np.nan),
        "away_sp_xfip":  away.get("sp_xfip",  np.nan),
        "sp_xfip_diff":  _safe_diff(away.get("sp_xfip"),  home.get("sp_xfip")),
        "home_sp_pa":    home.get("sp_pa",    np.nan),
        "away_sp_pa":    away.get("sp_pa",    np.nan),
        "sp_pa_diff":    _safe_diff(home.get("sp_pa"),    away.get("sp_pa")),
        "home_prior_win_pct": home.get("prior_win_pct", np.nan),
        "away_prior_win_pct": away.get("prior_win_pct", np.nan),
        "prior_win_pct_diff": _safe_diff(home.get("prior_win_pct"), away.get("prior_win_pct")),
        # Current-season running win% + delta vs prior-year baseline
        "home_season_win_pct":       home.get("season_win_pct", np.nan),
        "away_season_win_pct":       away.get("season_win_pct", np.nan),
        "season_win_pct_diff":       _safe_diff(home.get("season_win_pct"), away.get("season_win_pct")),
        "home_season_win_pct_delta": home.get("season_win_pct_delta", np.nan),
        "away_season_win_pct_delta": away.get("season_win_pct_delta", np.nan),
        "season_win_pct_delta_diff": _safe_diff(home.get("season_win_pct_delta"), away.get("season_win_pct_delta")),
        # SP workload: days since last start + outs in previous outing
        "home_sp_days_rest":  home.get("sp_days_rest", np.nan),
        "away_sp_days_rest":  away.get("sp_days_rest", np.nan),
        "sp_days_rest_diff":  _safe_diff(home.get("sp_days_rest"), away.get("sp_days_rest")),
        "home_sp_outs_last":  home.get("sp_outs_last", np.nan),
        "away_sp_outs_last":  away.get("sp_outs_last", np.nan),
        "sp_outs_last_diff":  _safe_diff(home.get("sp_outs_last"), away.get("sp_outs_last")),
        # Prior-season Statcast power metrics
        "home_barrel_pct":   home.get("barrel_pct",   np.nan),
        "away_barrel_pct":   away.get("barrel_pct",   np.nan),
        "barrel_pct_diff":   _safe_diff(home.get("barrel_pct"),   away.get("barrel_pct")),
        "home_hard_hit_pct": home.get("hard_hit_pct", np.nan),
        "away_hard_hit_pct": away.get("hard_hit_pct", np.nan),
        "hard_hit_pct_diff": _safe_diff(home.get("hard_hit_pct"), away.get("hard_hit_pct")),
        # Vegas consensus probability (devigged moneyline)
        "vegas_home_prob": home.get("vegas_home_prob", np.nan),
        # Confirmed lineup average OPS
        "home_lineup_ops":    home.get("lineup_ops",   np.nan),
        "away_lineup_ops":    away.get("lineup_ops",   np.nan),
        "lineup_ops_diff":    _safe_diff(home.get("lineup_ops"), away.get("lineup_ops")),
        "home_lineup_ops_vs_sp": home.get("lineup_ops_vs_sp", np.nan),
        "away_lineup_ops_vs_sp": away.get("lineup_ops_vs_sp", np.nan),
        "lineup_ops_vs_sp_diff": _safe_diff(home.get("lineup_ops_vs_sp"),
                                             away.get("lineup_ops_vs_sp")),
        "home_lineup_known_batters": home.get("lineup_known_batters", np.nan),
        "away_lineup_known_batters": away.get("lineup_known_batters", np.nan),
        "lineup_known_batters_diff": _safe_diff(home.get("lineup_known_batters"),
                                                 away.get("lineup_known_batters")),
        # Batting OPS vs SP handedness
        "home_batting_ops_vs_sp":  home.get("batting_ops_vs_sp", np.nan),
        "away_batting_ops_vs_sp":  away.get("batting_ops_vs_sp", np.nan),
        "batting_ops_vs_sp_diff":  _safe_diff(home.get("batting_ops_vs_sp"),
                                               away.get("batting_ops_vs_sp")),
        "home_platoon_advantage":  home.get("platoon_advantage", np.nan),
        "away_platoon_advantage":  away.get("platoon_advantage", np.nan),
        "platoon_advantage_diff":  _safe_diff(home.get("platoon_advantage"),
                                               away.get("platoon_advantage")),
        "home_opp_sp_is_lhp":      home.get("opp_sp_is_lhp", np.nan),
        "away_opp_sp_is_lhp":      away.get("opp_sp_is_lhp", np.nan),
        "home_sp_is_lhp":          home.get("sp_is_lhp", np.nan),
        "away_sp_is_lhp":          away.get("sp_is_lhp", np.nan),
        "both_sp_same_hand":       home.get("both_sp_same_hand", np.nan),
    }])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_matchup(home_team: str,
                    away_team: str,
                    year: int,
                    game_date: str | None = None,
                    ump_name: str | None = None,
                    home_sp_name: str | None = None,
                    away_sp_name: str | None = None,
                    verbose: bool = True,
                    model_mode: str = "independent") -> float:
    home_team = canonical_team(home_team)
    away_team = canonical_team(away_team)

    def _is_unknown_sp(name: str | None) -> bool:
        if name is None:
            return False
        return str(name).strip().lower() in {"", "tbd", "tb/a", "to be determined", "unknown"}

    home_sp_unknown = _is_unknown_sp(home_sp_name)
    away_sp_unknown = _is_unknown_sp(away_sp_name)
    if home_sp_unknown:
        home_sp_name = None
    if away_sp_unknown:
        away_sp_name = None

    blend_model_path = os.path.join(MODEL_DIR, "win_prob_blend_trees.pkl")
    independent_model_path = os.path.join(MODEL_DIR, "win_prob_model.pkl")
    market_model_path = os.path.join(MODEL_DIR, "win_prob_market_model.pkl")
    if not os.path.exists(blend_model_path) and not os.path.exists(independent_model_path):
        raise FileNotFoundError(
            f"Model not found at {blend_model_path} or {independent_model_path}. "
            "Run train_pure_baseball_ensemble.py first."
        )

    valid_model_modes = {"auto", "independent", "market"}
    if model_mode not in valid_model_modes:
        raise ValueError(
            f"model_mode must be one of {sorted(valid_model_modes)}, got {model_mode!r}"
        )

    features_df = _load_table_or_csv("features", "features.csv", parse_dates=["Date"])
    live_logs   = _load_prediction_game_logs()

    # Resolve game date (default = today)
    gdate = pd.Timestamp(game_date) if game_date else pd.Timestamp.today().normalize()

    home_feats = get_latest_team_features(home_team, year, features_df, live_logs)
    away_feats = get_latest_team_features(away_team, year, features_df, live_logs)
    home_feats["game_month"] = float(gdate.month)
    home_feats["game_day_of_year"] = float(gdate.dayofyear)
    home_feats["is_early_season"] = 1.0 if gdate.month in (3, 4) else 0.0
    home_feats["team"] = home_team
    away_feats["team"] = away_team

    # Compute rest & travel live from schedule history
    home_rt = compute_rest_travel(home_team, gdate, home_team, features_df)
    away_rt = compute_rest_travel(away_team, gdate, home_team, features_df)
    home_feats.update(home_rt)
    away_feats.update(away_rt)

    # Compute head-to-head history live
    h2h = compute_h2h_live(home_team, away_team, gdate, features_df)
    home_feats.update(h2h)

    # Weather at game time
    try:
        weather_df = _load_weather_data()
        wx = get_game_weather(home_team, str(gdate.date()), weather_df) if weather_df is not None else {}
    except Exception:
        wx = {}
    if wx:
        home_feats["temp_f"]         = wx.get("temp_f", np.nan)
        home_feats["wind_speed_mph"] = wx.get("wind_speed_mph", np.nan)
        home_feats["wind_dir_deg"]   = wx.get("wind_dir_deg", np.nan)
        home_feats["humidity_pct"]   = wx.get("humidity_pct", np.nan)
        home_feats["wind_to_cf"]     = wind_to_cf(
            wx.get("wind_speed_mph", 0.0),
            wx.get("wind_dir_deg",   0.0),
            home_team,
        )
        # Dome override
        if home_team in FULL_DOME:
            home_feats.update({"temp_f": 72.0, "wind_speed_mph": 0.0,
                                "wind_to_cf": 0.0, "humidity_pct": 50.0})

    # Prior-season team batting OPS (use year-1 to match training logic)
    bat_path = os.path.join(DATA_DIR, "batting_stats.csv")
    try:
        batting_df = _load_table_or_csv("batting_stats", "batting_stats.csv")
        home_ops = get_team_ops(home_team, year - 1, batting_df) if batting_df is not None else np.nan
        away_ops = get_team_ops(away_team, year - 1, batting_df) if batting_df is not None else np.nan
    except Exception:
        home_ops = away_ops = np.nan
    home_feats["team_ops"] = home_ops
    away_feats["team_ops"] = away_ops

    # Prior-season Statcast batting (barrel rate, hard hit%)
    try:
        statcast_df = _load_table_or_csv("statcast_batting", "statcast_batting.csv")
        home_sc = get_team_statcast(home_team, year - 1, statcast_df)
        away_sc = get_team_statcast(away_team, year - 1, statcast_df)
    except Exception:
        home_sc = away_sc = {"barrel_pct": np.nan, "hard_hit_pct": np.nan}
    home_feats["barrel_pct"]   = home_sc.get("barrel_pct",   np.nan)
    home_feats["hard_hit_pct"] = home_sc.get("hard_hit_pct", np.nan)
    away_feats["barrel_pct"]   = away_sc.get("barrel_pct",   np.nan)
    away_feats["hard_hit_pct"] = away_sc.get("hard_hit_pct", np.nan)

    # Compute bullpen usage (outs in last 3 days) from Retrosheet + live logs
    ump_path     = os.path.join(DATA_DIR, "umpire_game_logs.csv")
    try:
        game_logs_df = _load_pitcher_game_logs()
        if game_logs_df is not None and not game_logs_df.empty:
            home_bu = compute_bullpen_usage_live(home_team, gdate, game_logs_df)
            away_bu = compute_bullpen_usage_live(away_team, gdate, game_logs_df)
        else:
            game_logs_df = pd.DataFrame()
            home_bu = away_bu = np.nan
    except Exception:
        game_logs_df = pd.DataFrame()
        home_bu = away_bu = np.nan
    home_feats["bullpen_outs_3d"] = home_bu
    away_feats["bullpen_outs_3d"] = away_bu

    # IL counts — live API for recent games (cached per day), snapshot for history
    il_snapshot = _load_table_or_csv("il_counts", "il_counts.csv", parse_dates=["date"])
    home_il = _il_from_snapshot(home_team, gdate, il_snapshot)
    away_il = _il_from_snapshot(away_team, gdate, il_snapshot)
    home_feats["il_count"] = home_il
    away_feats["il_count"] = away_il

    # IL quality — WAR-weighted score for players currently on IL
    war_path = os.path.join(DATA_DIR, "player_war.csv")
    try:
        war_df = _load_table_or_csv("player_war", "player_war.csv")
        if war_df is not None and not war_df.empty:
            # For recent games use live API; for historical fall back to il_quality.csv snapshot
            days_ago = (pd.Timestamp.today().normalize() - gdate).days
            if days_ago <= _LIVE_IL_DAYS:
                home_il_war = get_current_il_quality(home_team, str(gdate.date()), war_df)
                away_il_war = get_current_il_quality(away_team, str(gdate.date()), war_df)
            else:
                il_quality_df = _load_table_or_csv("il_quality", "il_quality.csv",
                                                   parse_dates=["date"])
                def _war_from_snapshot(team: str) -> float:
                    if il_quality_df is None or il_quality_df.empty:
                        return np.nan
                    t_df = il_quality_df[il_quality_df["team"] == team].sort_values("date")
                    past = t_df[t_df["date"] <= gdate]
                    return float(past.iloc[-1]["il_war_score"]) if not past.empty else np.nan
                home_il_war = _war_from_snapshot(home_team)
                away_il_war = _war_from_snapshot(away_team)
        else:
            home_il_war = away_il_war = np.nan
    except Exception:
        home_il_war = away_il_war = np.nan
    home_feats["il_war"] = home_il_war
    away_feats["il_war"] = away_il_war

    # Umpire run factor
    ump_logs_df = _load_db_table("umpire_game_logs", parse_dates=["game_date"])
    if ump_logs_df is None or ump_logs_df.empty:
        ump_logs_df = _load_csv(ump_path, parse_dates=["game_date"]) if os.path.exists(ump_path) else None
    ump_factor = get_ump_run_factor(ump_name, gdate, game_logs_df, ump_logs_df)
    home_feats["ump_run_factor"] = ump_factor

    # SP pitch stuff — FBv, SwStr%, K%, xFIP, handedness
    home_sp_throws = away_sp_throws = None
    try:
        pitcher_stuff_df = _load_pitcher_stuff_data()
        # For live/future predictions, use current-season SP stuff when the
        # pitcher has enough PA; get_pitcher_stuff blends small samples with
        # prior-year data. Historical training still uses prior-season values
        # to avoid leakage.
        stuff_year = year
        empty_stuff = {
            "FBv": np.nan, "SwStr_pct": np.nan, "K_pct": np.nan,
            "xFIP": np.nan, "Throws": None, "GS": np.nan, "pa": np.nan,
        }
        # Prefer explicitly supplied SP names; fall back to last known from features.
        # A literal TBD is different from "not supplied": keep it neutral instead
        # of using a team median or a stale recent starter.
        if home_sp_unknown:
            home_sp_norm = None
            home_stuff = empty_stuff.copy()
        elif home_sp_name:
            home_sp_norm = _normalize_name(home_sp_name)
            home_stuff = get_pitcher_stuff(home_sp_norm or "", home_team, stuff_year,
                                           pitcher_stuff_df)
        else:
            home_sp_norm = _get_recent_sp_name(home_team, gdate, features_df, side="home")
            home_stuff = get_pitcher_stuff(home_sp_norm or "", home_team, stuff_year,
                                           pitcher_stuff_df)
        if away_sp_unknown:
            away_sp_norm = None
            away_stuff = empty_stuff.copy()
        elif away_sp_name:
            away_sp_norm = _normalize_name(away_sp_name)
            away_stuff = get_pitcher_stuff(away_sp_norm or "", away_team, stuff_year,
                                           pitcher_stuff_df)
        else:
            away_sp_norm = _get_recent_sp_name(away_team, gdate, features_df, side="away")
            away_stuff = get_pitcher_stuff(away_sp_norm or "", away_team, stuff_year,
                                           pitcher_stuff_df)
        home_feats.update({k: v for k, v in {
            "sp_fbv": home_stuff["FBv"], "sp_swstr": home_stuff["SwStr_pct"],
            "sp_k_pct": home_stuff["K_pct"], "sp_xfip": home_stuff["xFIP"],
            "sp_pa": home_stuff.get("pa", np.nan),
        }.items()})
        away_feats.update({k: v for k, v in {
            "sp_fbv": away_stuff["FBv"], "sp_swstr": away_stuff["SwStr_pct"],
            "sp_k_pct": away_stuff["K_pct"], "sp_xfip": away_stuff["xFIP"],
            "sp_pa": away_stuff.get("pa", np.nan),
        }.items()})
        home_sp_throws = home_stuff.get("Throws")
        away_sp_throws = away_stuff.get("Throws")

        # Override team-level sp_era_adj with the named starter's xFIP when available.
        # sp_era_adj from features.csv reflects the last game's starter, not today's.
        # xFIP is on the same ERA scale and is a better per-pitcher signal.
        if pd.notna(home_stuff.get("xFIP")):
            home_feats["sp_era"] = home_stuff["xFIP"]
        if pd.notna(away_stuff.get("xFIP")):
            away_feats["sp_era"] = away_stuff["xFIP"]
    except Exception:
        pass

    # SP workload: days since last start + outs in last start
    try:
        home_sp_work = compute_sp_workload_live(
            home_sp_norm, None, gdate, game_logs_df)
        away_sp_work = compute_sp_workload_live(
            away_sp_norm, None, gdate, game_logs_df)
    except Exception:
        home_sp_work = away_sp_work = {"sp_days_rest": np.nan, "sp_outs_last": np.nan}
    home_feats["sp_days_rest"] = home_sp_work["sp_days_rest"]
    home_feats["sp_outs_last"] = home_sp_work["sp_outs_last"]
    away_feats["sp_days_rest"] = away_sp_work["sp_days_rest"]
    away_feats["sp_outs_last"] = away_sp_work["sp_outs_last"]

    # L/R batting splits (OPS vs SP handedness)
    try:
        splits_df = _load_table_or_csv("team_splits", "team_splits.csv")
        home_feats["batting_ops_vs_sp"] = get_team_split_ops(
            home_team, year, away_sp_throws, splits_df)
        away_feats["batting_ops_vs_sp"] = get_team_split_ops(
            away_team, year, home_sp_throws, splits_df)
        home_feats["platoon_advantage"] = _safe_diff(
            home_feats.get("batting_ops_vs_sp"), home_feats.get("team_ops")
        )
        away_feats["platoon_advantage"] = _safe_diff(
            away_feats.get("batting_ops_vs_sp"), away_feats.get("team_ops")
        )
    except Exception:
        pass

    home_feats["opp_sp_is_lhp"] = 1.0 if away_sp_throws == "L" else 0.0 if away_sp_throws == "R" else np.nan
    away_feats["opp_sp_is_lhp"] = 1.0 if home_sp_throws == "L" else 0.0 if home_sp_throws == "R" else np.nan
    home_feats["sp_is_lhp"] = 1.0 if home_sp_throws == "L" else 0.0 if home_sp_throws == "R" else np.nan
    away_feats["sp_is_lhp"] = 1.0 if away_sp_throws == "L" else 0.0 if away_sp_throws == "R" else np.nan
    if home_sp_throws in ("L", "R") and away_sp_throws in ("L", "R"):
        same_hand = 1.0 if home_sp_throws == away_sp_throws else 0.0
    else:
        same_hand = np.nan
    home_feats["both_sp_same_hand"] = same_hand
    away_feats["both_sp_same_hand"] = same_hand

    # Vegas moneylines (devigged consensus probability)
    try:
        odds_df = load_or_fetch_odds(game_date=str(gdate.date()))
        vegas_prob = get_home_implied_prob(home_team, away_team, odds_df)
        home_feats["vegas_home_prob"] = vegas_prob
    except Exception:
        odds_df = None
        vegas_prob = np.nan

    # Confirmed lineup OPS
    try:
        lineups_df = fetch_confirmed_lineups(game_date=str(gdate.date()))
        h_lops, a_lops = get_lineup_ops(home_team, away_team, lineups_df)
        home_feats["lineup_ops"] = h_lops
        away_feats["lineup_ops"] = a_lops
        # Live lineup fetch currently returns season OPS, not player L/R splits.
        # Use it as a live fallback for the trained lineup-vs-SP slot.
        home_feats["lineup_ops_vs_sp"] = h_lops
        away_feats["lineup_ops_vs_sp"] = a_lops
        home_feats["lineup_known_batters"] = 9 if pd.notna(h_lops) else np.nan
        away_feats["lineup_known_batters"] = 9 if pd.notna(a_lops) else np.nan
    except Exception:
        lineups_df = None
        h_lops = a_lops = np.nan

    use_market_model = (
        model_mode == "market"
    )
    model_path = market_model_path if use_market_model else (
        blend_model_path if os.path.exists(blend_model_path) else independent_model_path
    )
    if use_market_model and not os.path.exists(market_model_path):
        raise FileNotFoundError(
            f"Market-aware model not found at {market_model_path}. Run train_model.py first."
        )

    artifact = _load_model(model_path)
    feat_cols = artifact["features"]

    X = build_input_row(home_feats, away_feats)
    for col in feat_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = apply_feature_defaults(X, feat_cols)
    X = X[feat_cols]
    if "pipelines" in artifact:
        model_probs = {
            name: float(pipeline.predict_proba(X)[0, 1])
            for name, pipeline in artifact["pipelines"].items()
        }
        blend_weights = artifact.get("blend_weights") or {}
        if blend_weights:
            total_weight = sum(float(blend_weights.get(name, 0.0)) for name in model_probs)
            if total_weight > 0:
                prob_home_win = float(
                    sum(
                        model_probs[name] * float(blend_weights.get(name, 0.0))
                        for name in model_probs
                    ) / total_weight
                )
            else:
                prob_home_win = float(np.mean(list(model_probs.values())))
        else:
            prob_home_win = float(np.mean(list(model_probs.values())))
    else:
        pipeline = artifact["pipeline"]
        model_probs = {}
        prob_home_win = float(pipeline.predict_proba(X)[0, 1])

    if verbose:
        print(f"\n{'='*45}")
        print(f"  Matchup:  {away_team}  @  {home_team}")
        print(f"{'='*45}")
        print(f"  Home ({home_team}) win probability:  {prob_home_win:.1%}")
        print(f"  Away ({away_team}) win probability:  {1 - prob_home_win:.1%}")
        print(f"{'='*45}")
        model_label = artifact.get("model_type", "market_aware" if use_market_model else "market_independent")
        print(f"  Probability model        : {model_label}")
        if model_probs:
            blend_bits = ", ".join(f"{name} {prob:.1%}" for name, prob in model_probs.items())
            print(f"  Blend components         : {blend_bits}")
        print(f"\n  Feature snapshot (last {home_team} data):")
        print(f"    Home rolling run-diff  : {home_feats['rolling_rd']:+.2f}")
        print(f"    Away rolling run-diff  : {away_feats['rolling_rd']:+.2f}")
        print(f"    Home runs/game (15g)   : {home_feats['rolling_rs']:.2f}")
        print(f"    Away runs/game (15g)   : {away_feats['rolling_rs']:.2f}")
        if not np.isnan(home_feats["sp_era"]):
            print(f"    Home SP ERA (adj)         : {home_feats['sp_era']:.2f}")
            print(f"    Away SP ERA (adj)         : {away_feats['sp_era']:.2f}")
        if not np.isnan(home_feats["sp_inseason_era"]):
            print(f"    Home SP in-season ERA     : {home_feats['sp_inseason_era']:.2f}")
            print(f"    Away SP in-season ERA     : {away_feats['sp_inseason_era']:.2f}")
        if not np.isnan(home_feats["bullpen_inseason_era"]):
            print(f"    Home bullpen in-season ERA: {home_feats['bullpen_inseason_era']:.2f}")
            print(f"    Away bullpen in-season ERA: {away_feats['bullpen_inseason_era']:.2f}")
        if not np.isnan(home_feats["park_factor"]):
            print(f"    Park factor            : {home_feats['park_factor']:.3f}")
        hr = home_feats.get("days_rest")
        ar = away_feats.get("days_rest")
        if hr == hr:   # not NaN
            print(f"    Home days rest         : {int(hr)}")
            print(f"    Away days rest         : {int(ar)}")
        at = away_feats.get("travel_miles")
        if at == at:
            print(f"    Away travel miles      : {at:,.0f} mi")
        h2h_wr = home_feats.get("h2h_home_win_rate")
        if h2h_wr == h2h_wr:   # not NaN
            h2h_rd = home_feats.get("h2h_home_run_diff", np.nan)
            print(f"    H2H home win rate      : {h2h_wr:.1%}  (last 10 meetings, min 3)")
            print(f"    H2H run diff (home)    : {h2h_rd:+.2f} runs/game")
        hbu = home_feats.get("bullpen_outs_3d")
        abu = away_feats.get("bullpen_outs_3d")
        if hbu == hbu:   # not NaN
            print(f"    Home bullpen outs (3d) : {int(hbu)} outs  (~{hbu/3:.1f} IP)")
            print(f"    Away bullpen outs (3d) : {int(abu)} outs  (~{abu/3:.1f} IP)")
        hswp = home_feats.get("season_win_pct")
        aswp = away_feats.get("season_win_pct")
        hpwp = home_feats.get("prior_win_pct", np.nan)
        apwp = away_feats.get("prior_win_pct", np.nan)
        if hswp == hswp:   # not NaN
            hdelta = hswp - hpwp if not np.isnan(hpwp) else np.nan
            adelta = aswp - apwp if not np.isnan(apwp) else np.nan
            delta_str = lambda d: f" ({d:+.1%} vs prior yr)" if d == d else ""
            print(f"    Home season win%       : {hswp:.1%}{delta_str(hdelta)}")
            print(f"    Away season win%       : {aswp:.1%}{delta_str(adelta)}")
        hop = home_feats.get("team_ops")
        aop = away_feats.get("team_ops")
        if hop == hop:
            print(f"    Home team OPS (prior yr): {hop:.3f}")
            print(f"    Away team OPS (prior yr): {aop:.3f}")
        tf = home_feats.get("temp_f")
        if tf is not None and tf == tf:
            ws  = home_feats.get("wind_speed_mph", 0)
            wtc = home_feats.get("wind_to_cf", 0)
            hum = home_feats.get("humidity_pct", np.nan)
            direction = "out" if wtc > 0 else "in"
            print(f"    Temperature            : {tf:.0f}°F")
            print(f"    Wind                   : {ws:.0f} mph "
                  f"({'dome' if home_team in FULL_DOME else f'{abs(wtc):.1f} mph blowing {direction}'})")
            if hum == hum:
                print(f"    Humidity               : {hum:.0f}%")
        hil = home_feats.get("il_count", 0)
        ail = away_feats.get("il_count", 0)
        print(f"    Home IL count          : {hil} players")
        print(f"    Away IL count          : {ail} players")
        urf = home_feats.get("ump_run_factor", 1.0)
        if ump_name:
            tendency = ("hitter-friendly" if urf > 1.02
                        else "pitcher-friendly" if urf < 0.98
                        else "neutral")
            print(f"    Umpire                 : {ump_name}  "
                  f"(run factor {urf:.3f} — {tendency})")
        else:
            print(f"    Umpire                 : unknown (using neutral 1.000)")
        hdr = home_feats.get("sp_days_rest")
        if hdr == hdr:   # not NaN
            adr = away_feats.get("sp_days_rest", np.nan)
            hol = home_feats.get("sp_outs_last", np.nan)
            aol = away_feats.get("sp_outs_last", np.nan)
            hol_str = f"{hol/3:.1f} IP" if hol == hol else "?"
            aol_str = f"{aol/3:.1f} IP" if aol == aol else "?"
            print(f"    Home SP days rest      : {int(hdr)}d  (last start: {hol_str})")
            print(f"    Away SP days rest      : {int(adr)}d  (last start: {aol_str})")
        hfbv = home_feats.get("sp_fbv")
        if hfbv and hfbv == hfbv:
            afbv = away_feats.get("sp_fbv", np.nan)
            hxfip = home_feats.get("sp_xfip", np.nan)
            axfip = away_feats.get("sp_xfip", np.nan)
            hthrows = home_sp_throws or "?"
            athrows = away_sp_throws or "?"
            print(f"    Home SP FBv/xFIP ({hthrows})  : {hfbv:.1f} mph / {hxfip:.2f}" if hfbv==hfbv else "")
            print(f"    Away SP FBv/xFIP ({athrows})  : {afbv:.1f} mph / {axfip:.2f}" if afbv==afbv else "")
        hbsp = home_feats.get("batting_ops_vs_sp")
        if hbsp and hbsp == hbsp:
            absp = away_feats.get("batting_ops_vs_sp", np.nan)
            print(f"    Home OPS vs {away_sp_throws or '?'}HP  : {hbsp:.3f}")
            print(f"    Away OPS vs {home_sp_throws or '?'}HP  : {absp:.3f}")
        hlops = home_feats.get("lineup_ops")
        if hlops and hlops == hlops:
            alops = away_feats.get("lineup_ops", np.nan)
            print(f"    Home lineup OPS        : {hlops:.3f} (confirmed)")
            print(f"    Away lineup OPS        : {alops:.3f} (confirmed)")
        if not np.isnan(vegas_prob):
            ml_str = get_moneyline_str(home_team, away_team, odds_df)
            edge = prob_home_win - vegas_prob
            edge_str = f"{edge:+.1%}" if abs(edge) >= 0.01 else "~flat"
            print(f"    Vegas line (ML)        : {ml_str}")
            print(f"    Vegas home prob        : {vegas_prob:.1%}  (model edge: {edge_str})")

    return prob_home_win


# ---------------------------------------------------------------------------
# Batch predict from CSV
# ---------------------------------------------------------------------------

def predict_batch(input_csv: str, output_csv: str, year: int) -> None:
    """
    input_csv must have columns: home_team, away_team
    Output adds: home_win_prob, away_win_prob
    """
    matchups = pd.read_csv(input_csv)
    probs = []
    for _, row in matchups.iterrows():
        try:
            p = predict_matchup(row["home_team"], row["away_team"], year, verbose=False)
        except Exception as exc:
            print(f"  Warning: {row['home_team']} vs {row['away_team']} — {exc}")
            p = np.nan
        probs.append(p)

    matchups["home_win_prob"] = probs
    matchups["away_win_prob"] = 1 - matchups["home_win_prob"]
    matchups.to_csv(output_csv, index=False)
    print(f"Batch predictions saved to {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLB pre-game win probability",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Single matchup
    single = sub.add_parser("game", help="Predict a single game")
    single.add_argument("--home", required=True, help=f"Home team abbrev ({TEAM_ABBREV_HELP})")
    single.add_argument("--away", required=True, help="Away team abbrev")
    single.add_argument("--year", type=int, default=2026)
    single.add_argument("--date", default=None, help="Game date YYYY-MM-DD (default: today)")
    single.add_argument("--ump",  default=None, help="Home plate umpire name (optional)")
    single.add_argument(
        "--model-mode",
        choices=["auto", "independent", "market"],
        default="independent",
        help="Probability model to use. auto is an alias for independent.",
    )

    # Batch
    batch = sub.add_parser("batch", help="Predict from a CSV file")
    batch.add_argument("--input",  required=True, help="CSV with home_team, away_team columns")
    batch.add_argument("--output", required=True, help="Output CSV path")
    batch.add_argument("--year",   type=int, default=2025)

    args = parser.parse_args()

    if args.command == "game":
        predict_matchup(args.home.upper(), args.away.upper(), args.year,
                        game_date=args.date, ump_name=args.ump,
                        model_mode=args.model_mode)
    elif args.command == "batch":
        predict_batch(args.input, args.output, args.year)
    else:
        parser.print_help()
