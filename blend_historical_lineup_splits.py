"""
Blend prior-season and current-season-to-date player split OPS for lineups.

Input:
  data/historical_lineups.csv
  data/player_split_cache.json        # prior-season splits from fetch_historical_lineups.py

Output:
  Updates data/historical_lineups.csv with blended columns:
    home_lineup_blend_ops_vs_lhp, home_lineup_blend_ops_vs_rhp
    away_lineup_blend_ops_vs_lhp, away_lineup_blend_ops_vs_rhp

The current-season component is date-bounded through the day before each game,
so these features are pre-game safe.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from fetch_lineups import MLB_API_BASE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LINEUPS_PATH = os.path.join(DATA_DIR, "historical_lineups.csv")
PLAYER_SPLIT_CACHE_PATH = os.path.join(DATA_DIR, "player_split_cache.json")


def _parse_float(value) -> float:
    try:
        if value in (None, "", "-.--"):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _split_ids(value) -> list[int]:
    if pd.isna(value):
        return []
    ids = []
    for part in str(value).split():
        try:
            ids.append(int(part))
        except Exception:
            continue
    return ids[:9]


def _season_start(year: int) -> str:
    # Includes international/opening games without pulling spring training.
    return f"{year}-02-15"


def _fetch_bulk_player_split(year: int,
                             end_date: str,
                             sit_code: str) -> dict[int, tuple[float, int]]:
    url = (
        f"{MLB_API_BASE}/stats?stats=statSplits&group=hitting&season={year}"
        f"&sitCodes={sit_code}&sportIds=1&playerPool=ALL&limit=10000"
        f"&startDate={_season_start(year)}&endDate={end_date}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    out: dict[int, tuple[float, int]] = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        player = split.get("player") or {}
        stat = split.get("stat") or {}
        pid = player.get("id")
        if not pid:
            continue
        out[int(pid)] = (
            _parse_float(stat.get("ops")),
            int(stat.get("plateAppearances", 0) or 0),
        )
    return out


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    pairs = [
        (float(v), max(int(w), 1))
        for v, w in zip(values, weights)
        if v is not None and not pd.isna(v)
    ]
    if not pairs:
        return np.nan
    vals, wgts = zip(*pairs)
    return float(np.average(vals, weights=wgts))


def _prior_split(cache: dict, person_id: int, year: int, hand: str) -> tuple[float, int]:
    row = cache.get(f"{person_id}:{year - 1}") or {}
    if hand == "L":
        return row.get("ops_vs_lhp"), int(row.get("pa_vs_lhp", 0) or 0)
    return row.get("ops_vs_rhp"), int(row.get("pa_vs_rhp", 0) or 0)


def _current_split(snapshots: dict[str, dict[int, tuple[float, int]]],
                   person_id: int,
                   hand: str) -> tuple[float, int]:
    snap = snapshots["vl" if hand == "L" else "vr"]
    return snap.get(person_id, (np.nan, 0))


def _blend_player(prior_ops,
                  prior_pa: int,
                  current_ops,
                  current_pa: int,
                  prior_pa_cap: int) -> tuple[float, int]:
    p_ops = np.nan if prior_ops is None else float(prior_ops)
    c_ops = np.nan if current_ops is None else float(current_ops)
    p_w = min(max(int(prior_pa or 0), 0), prior_pa_cap)
    c_w = max(int(current_pa or 0), 0)

    vals, wgts = [], []
    if pd.notna(p_ops) and p_w > 0:
        vals.append(p_ops)
        wgts.append(p_w)
    if pd.notna(c_ops) and c_w > 0:
        vals.append(c_ops)
        wgts.append(c_w)
    if not vals:
        return np.nan, 0
    return float(np.average(vals, weights=wgts)), int(sum(wgts))


def _lineup_blend(ids: list[int],
                  year: int,
                  hand: str,
                  prior_cache: dict,
                  current_snapshots: dict[str, dict[int, tuple[float, int]]],
                  prior_pa_cap: int) -> tuple[float, int]:
    ops_values, weights = [], []
    for pid in ids:
        p_ops, p_pa = _prior_split(prior_cache, pid, year, hand)
        c_ops, c_pa = _current_split(current_snapshots, pid, hand)
        b_ops, b_w = _blend_player(p_ops, p_pa, c_ops, c_pa, prior_pa_cap)
        ops_values.append(b_ops)
        weights.append(b_w)
    return _weighted_mean(ops_values, weights), len(ids)


def blend_lineups(start_year: int | None = None,
                  end_year: int | None = None,
                  prior_pa_cap: int = 200,
                  flush_every_dates: int = 20) -> pd.DataFrame:
    df = pd.read_csv(LINEUPS_PATH, parse_dates=["game_date"])
    if start_year is not None:
        df = df[df["game_date"].dt.year >= start_year].copy()
    if end_year is not None:
        df = df[df["game_date"].dt.year <= end_year].copy()

    full = pd.read_csv(LINEUPS_PATH, parse_dates=["game_date"])
    with open(PLAYER_SPLIT_CACHE_PATH) as f:
        prior_cache = json.load(f)

    if df.empty:
        return full

    for col in [
        "home_lineup_blend_ops_vs_lhp", "home_lineup_blend_ops_vs_rhp",
        "away_lineup_blend_ops_vs_lhp", "away_lineup_blend_ops_vs_rhp",
    ]:
        if col not in full.columns:
            full[col] = np.nan

    dates = sorted(df["game_date"].dt.date.unique())
    processed = 0
    for game_date in dates:
        year = game_date.year
        end_dt = (pd.Timestamp(game_date) - timedelta(days=1)).date().isoformat()
        snapshots = {
            "vl": _fetch_bulk_player_split(year, end_dt, "vl"),
            "vr": _fetch_bulk_player_split(year, end_dt, "vr"),
        }

        date_mask = full["game_date"].dt.date == game_date
        for idx, row in full.loc[date_mask].iterrows():
            home_ids = _split_ids(row.get("home_lineup_ids"))
            away_ids = _split_ids(row.get("away_lineup_ids"))
            for hand, suffix in [("L", "lhp"), ("R", "rhp")]:
                home_ops, _ = _lineup_blend(
                    home_ids, year, hand, prior_cache, snapshots, prior_pa_cap
                )
                away_ops, _ = _lineup_blend(
                    away_ids, year, hand, prior_cache, snapshots, prior_pa_cap
                )
                full.at[idx, f"home_lineup_blend_ops_vs_{suffix}"] = home_ops
                full.at[idx, f"away_lineup_blend_ops_vs_{suffix}"] = away_ops

        processed += 1
        if processed % flush_every_dates == 0:
            full.to_csv(LINEUPS_PATH, index=False)
            print(f"  Blended through {game_date}; dates {processed}/{len(dates)}", flush=True)

    full.to_csv(LINEUPS_PATH, index=False)
    print(f"Saved blended lineup columns -> {LINEUPS_PATH}", flush=True)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend historical lineup split OPS")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--prior-pa-cap", type=int, default=200)
    parser.add_argument("--flush-every-dates", type=int, default=20)
    args = parser.parse_args()

    blend_lineups(
        start_year=args.start_year,
        end_year=args.end_year,
        prior_pa_cap=args.prior_pa_cap,
        flush_every_dates=max(1, args.flush_every_dates),
    )


if __name__ == "__main__":
    main()
