"""
Fetch historical hourly weather data for all 30 MLB parks from
Open-Meteo's free archive API (no API key required).

For each park we pull the full 2014-2026 season window in a single request,
extract the 7 pm local hour (representative game-time weather), and save to
data/weather.csv.

Output columns:
  date, home_team, temp_f, wind_speed_mph, wind_dir_deg, humidity_pct

Run this ONCE; subsequent calls load from cache unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_PATH = os.path.join(DATA_DIR, "weather.csv")

# ── Park locations ────────────────────────────────────────────────────────────
PARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4453, -112.0667),
    "ATL": (33.8908,  -84.4677),
    "BAL": (39.2838,  -76.6218),
    "BOS": (42.3467,  -71.0972),
    "CHC": (41.9484,  -87.6553),
    "CHW": (41.8300,  -87.6338),
    "CIN": (39.0979,  -84.5082),
    "CLE": (41.4962,  -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390,  -83.0485),
    "HOU": (29.7573,  -95.3555),
    "KCR": (39.0517,  -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781,  -80.2197),
    "MIL": (43.0280,  -87.9712),
    "MIN": (44.9817,  -93.2781),
    "NYM": (40.7571,  -73.8458),
    "NYY": (40.8296,  -73.9262),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9061,  -75.1665),
    "PIT": (40.4469,  -80.0057),
    "SDP": (32.7076, -117.1570),
    "SEA": (47.5914, -122.3325),
    "SFG": (37.7786, -122.3893),
    "STL": (38.6226,  -90.1928),
    "TBR": (27.7683,  -82.6534),
    "TEX": (32.7473,  -97.0831),
    "TOR": (43.6414,  -79.3894),
    "WSN": (38.8730,  -77.0074),
}

# Compass bearing from home plate to center field (degrees clockwise from N).
# Used to compute the wind component blowing TOWARD center field.
# Positive wind_to_cf → wind blowing out (hitter-friendly).
# Negative wind_to_cf → wind blowing in (pitcher-friendly).
PARK_CF_BEARING: dict[str, float] = {
    "ARI": 310, "ATL":  25, "BAL":  50, "BOS":  95,
    "CHC": 100, "CHW":  10, "CIN": 345, "CLE": 310,
    "COL":  30, "DET": 195, "HOU":  30, "KCR":  10,
    "LAA":  10, "LAD":   0, "MIA": 320, "MIL":   0,
    "MIN":  30, "NYM": 270, "NYY":  20, "OAK": 315,
    "PHI":  25, "PIT":  25, "SDP": 315, "SEA":  25,
    "SFG":  70, "STL": 330, "TBR":   0, "TEX":  30,
    "TOR": 295, "WSN": 355,
}

# Full domes — weather irrelevant; we'll override to neutral values
FULL_DOME: set[str] = {"TBR"}

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GAME_HOUR    = 19   # 7 pm local — representative start time


# ── API helpers ───────────────────────────────────────────────────────────────

def _fetch_open_meteo(lat: float, lon: float,
                      start: str, end: str,
                      forecast: bool = False,
                      max_retries: int = 5) -> dict:
    """
    Fetch hourly weather from Open-Meteo for a single location + date range.
    Uses archive API for historical data, forecast API for upcoming games.
    Retries on 429 with exponential backoff.
    """
    url    = FORECAST_URL if forecast else ARCHIVE_URL
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start,
        "end_date":         end,
        "hourly":           ("temperature_2m,windspeed_10m,"
                             "winddirection_10m,relativehumidity_2m"),
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "timezone":         "auto",
    }
    wait = 30  # initial backoff seconds
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            if attempt < max_retries - 1:
                print(f"429 — waiting {wait}s before retry {attempt + 1}/{max_retries - 1}...",
                      end=" ", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 300)  # exponential backoff, cap at 5 min
                continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()  # final attempt already failed above
    return r.json()


def _parse_hourly(raw: dict, team: str) -> pd.DataFrame:
    """
    Extract the 7 pm row for each calendar date from an Open-Meteo response.
    """
    df = pd.DataFrame({
        "datetime":      pd.to_datetime(raw["hourly"]["time"]),
        "temp_f":        raw["hourly"]["temperature_2m"],
        "wind_speed_mph": raw["hourly"]["windspeed_10m"],
        "wind_dir_deg":  raw["hourly"]["winddirection_10m"],
        "humidity_pct":  raw["hourly"]["relativehumidity_2m"],
    })
    df["home_team"] = team
    df = df[df["datetime"].dt.hour == GAME_HOUR].copy()
    df["date"] = df["datetime"].dt.date
    return df[["date", "home_team",
               "temp_f", "wind_speed_mph", "wind_dir_deg", "humidity_pct"]]


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_all_weather(start_year: int = 2014,
                      end_year:   int = 2026,
                      cache_path: str = CACHE_PATH,
                      refresh:    bool = False) -> pd.DataFrame:
    """
    Fetch weather for all parks over start_year–end_year and cache results.
    Each park is a single API call covering the full date range.

    Incremental by park: if cache exists, only re-fetches teams missing from it
    (unless refresh=True, which re-fetches all 30).
    """
    existing: pd.DataFrame | None = None
    already_fetched: set[str] = set()

    if os.path.exists(cache_path) and not refresh:
        existing = pd.read_csv(cache_path, parse_dates=["date"])
        already_fetched = set(existing["home_team"].unique())
        missing = [t for t in PARK_COORDS if t not in already_fetched]
        if not missing:
            print(f"Loading cached weather data from {cache_path}")
            print(f"  {len(existing):,} rows covering "
                  f"{existing['date'].min().date()} → {existing['date'].max().date()}")
            return existing
        print(f"Cache has {len(already_fetched)}/30 parks; fetching {len(missing)} missing: {missing}")

    start_str = f"{start_year}-03-01"
    # Archive API only accepts dates up to ~5 days ago; cap at 2 days ago
    archive_end = min(
        pd.Timestamp(f"{end_year}-11-15"),
        pd.Timestamp.today().normalize() - pd.Timedelta(days=2),
    )
    end_str = archive_end.strftime("%Y-%m-%d")

    teams_to_fetch = {t: c for t, c in PARK_COORDS.items()
                      if refresh or t not in already_fetched}
    total = len(teams_to_fetch)
    print(f"Fetching weather {start_str} → {end_str} for {total} parks...")

    frames = []
    for i, (team, (lat, lon)) in enumerate(teams_to_fetch.items(), 1):
        print(f"  [{i:02d}/{total:02d}] {team}...", end=" ", flush=True)
        try:
            raw = _fetch_open_meteo(lat, lon, start_str, end_str)
            df  = _parse_hourly(raw, team)
            frames.append(df)
            print(f"{len(df):,} rows")
        except Exception as exc:
            print(f"FAILED ({exc})")
        if i < total:
            time.sleep(5.0)   # conservative delay to avoid 429 rate limiting

    # Merge newly fetched data with any previously cached parks
    all_frames = []
    if existing is not None and not refresh:
        all_frames.append(existing[existing["home_team"].isin(already_fetched)])
    all_frames.extend(frames)

    if not all_frames:
        raise RuntimeError("No weather data fetched — all parks failed.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(cache_path, index=False)
    print(f"\nSaved {len(combined):,} weather rows to {cache_path}")
    return combined


def refresh_recent_weather(cache_path: str = CACHE_PATH,
                           stale_days: int = 3) -> pd.DataFrame:
    """
    Append weather for any dates missing from the cache.
    Safe to call daily — does nothing if cache is already current.
    """
    if not os.path.exists(cache_path):
        return fetch_all_weather(cache_path=cache_path)

    existing = pd.read_csv(cache_path)
    existing["date"] = pd.to_datetime(existing["date"], format="mixed")
    max_date = existing["date"].max()
    archive_end = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)

    if max_date >= archive_end - pd.Timedelta(days=stale_days):
        print(f"Weather cache is current (through {max_date.date()}).")
        return existing

    since = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = archive_end.strftime("%Y-%m-%d")
    print(f"Refreshing weather {since} → {end} for all parks...")

    frames = []
    parks  = list(PARK_COORDS.items())
    for i, (team, (lat, lon)) in enumerate(parks, 1):
        print(f"  [{i:02d}/{len(parks)}] {team}...", end=" ", flush=True)
        try:
            raw = _fetch_open_meteo(lat, lon, since, end)
            df  = _parse_hourly(raw, team)
            frames.append(df)
            print(f"{len(df)} rows")
        except Exception as exc:
            print(f"FAILED ({exc})")
        if i < len(parks):
            time.sleep(3.0)

    if not frames:
        print("No new weather rows fetched.")
        return existing

    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"])
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], format="mixed")
    combined = (combined.drop_duplicates(subset=["date", "home_team"])
                        .sort_values(["home_team", "date"])
                        .reset_index(drop=True))
    combined.to_csv(cache_path, index=False)
    print(f"Weather updated: {len(new_data)} new rows → {len(combined):,} total")
    return combined


# ── Live weather for predict.py ───────────────────────────────────────────────

def get_game_weather(home_team: str,
                     game_date: str,
                     weather_df: pd.DataFrame | None = None) -> dict:
    """
    Return weather dict for a specific park + date.
    First tries the cached weather_df; falls back to a live API call
    (uses forecast endpoint for dates within 7 days, archive otherwise).
    """
    if weather_df is not None:
        mask = (
            (weather_df["home_team"] == home_team)
            & (weather_df["date"].astype(str) == str(game_date))
        )
        row = weather_df[mask]
        if not row.empty:
            r = row.iloc[0]
            return {
                "temp_f":        float(r["temp_f"]),
                "wind_speed_mph": float(r["wind_speed_mph"]),
                "wind_dir_deg":  float(r["wind_dir_deg"]),
                "humidity_pct":  float(r["humidity_pct"]),
            }

    # Live fetch
    lat, lon = PARK_COORDS.get(home_team, (None, None))
    if lat is None:
        return {}
    today    = pd.Timestamp.today().normalize()
    gdate    = pd.Timestamp(game_date)
    forecast = (gdate - today).days >= -1    # within 7-day forecast window
    try:
        raw = _fetch_open_meteo(lat, lon, game_date, game_date, forecast=forecast)
        df  = _parse_hourly(raw, home_team)
        if df.empty:
            return {}
        r = df.iloc[0]
        return {
            "temp_f":        float(r["temp_f"]),
            "wind_speed_mph": float(r["wind_speed_mph"]),
            "wind_dir_deg":  float(r["wind_dir_deg"]),
            "humidity_pct":  float(r["humidity_pct"]),
        }
    except Exception:
        return {}


# ── Wind-to-CF computation ────────────────────────────────────────────────────

def wind_to_cf(wind_speed: float, wind_from_deg: float, team: str) -> float:
    """
    Component of wind blowing TOWARD center field (mph).
    Positive = blowing out (helps hitters); Negative = blowing in (helps pitchers).

    Wind direction convention: meteorological (degrees the wind is coming FROM).
    We convert to where the wind is going, then project onto the HP→CF axis.
    """
    if team in FULL_DOME:
        return 0.0
    cf_bearing = PARK_CF_BEARING.get(team, 0.0)
    # Direction the wind is blowing TOWARD
    wind_toward = (wind_from_deg + 180.0) % 360.0
    # Angle between wind direction and CF bearing
    delta = math.radians(wind_toward - cf_bearing)
    return wind_speed * math.cos(delta)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLB park weather data")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if cache exists")
    args = parser.parse_args()

    df = fetch_all_weather(refresh=args.refresh)
    print("\nSample (Wrigley Field / CHC):")
    print(df[df["home_team"] == "CHC"].head(5).to_string(index=False))
