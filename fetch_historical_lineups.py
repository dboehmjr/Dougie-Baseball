"""
Fetch historical MLB starting lineups and prior-season player split OPS.

The MLB Stats API exposes archived boxscores with battingOrder. For training,
we use those actual starting lineups and join each hitter to prior-season OPS
splits vs LHP/RHP. Prior-season splits avoid leaking future plate appearances
from the game season into pre-game features.

Output:
  data/historical_lineups.csv

Usage:
  python fetch_historical_lineups.py --start-year 2015 --end-year 2025
  python fetch_historical_lineups.py --start-date 2025-04-01 --end-date 2025-04-07
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from fetch_lineups import MLB_API_BASE, TEAM_IDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "historical_lineups.csv")
PLAYER_SPLIT_CACHE_PATH = os.path.join(DATA_DIR, "player_split_cache.json")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True)


def _get_json(url: str, timeout: int = 20) -> dict:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _schedule(start_date: str, end_date: str) -> list[dict]:
    url = (
        f"{MLB_API_BASE}/schedule?sportId=1"
        f"&startDate={start_date}&endDate={end_date}&hydrate=teams"
    )
    data = _get_json(url, timeout=30)
    games = []
    for entry in data.get("dates", []):
        for game in entry.get("games", []):
            status = (game.get("status") or {}).get("codedGameState")
            if status not in {"F", "O"}:
                continue
            home = game.get("teams", {}).get("home", {}).get("team", {})
            away = game.get("teams", {}).get("away", {}).get("team", {})
            games.append({
                "game_pk": int(game["gamePk"]),
                "game_date": game["officialDate"],
                "home_team": TEAM_IDS.get(home.get("id"), ""),
                "away_team": TEAM_IDS.get(away.get("id"), ""),
            })
    return games


def _boxscore_lineup_ids(game_pk: int) -> dict:
    data = _get_json(f"{MLB_API_BASE}/game/{game_pk}/boxscore", timeout=30)
    out = {
        "home_lineup_ids": [],
        "away_lineup_ids": [],
        "home_lineup_names": [],
        "away_lineup_names": [],
    }
    for side in ("home", "away"):
        team = data.get("teams", {}).get(side, {})
        batting_order = team.get("battingOrder", [])[:9]
        players = team.get("players", {})
        ids = []
        names = []
        for raw_id in batting_order:
            try:
                pid = int(raw_id)
            except Exception:
                continue
            player = players.get(f"ID{pid}", {}).get("person", {})
            ids.append(pid)
            names.append(player.get("fullName", ""))
        out[f"{side}_lineup_ids"] = ids
        out[f"{side}_lineup_names"] = names
    return out


def _parse_float(value) -> float:
    try:
        if value in (None, "", "-.--"):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _fetch_player_split(person_id: int, year: int, sit_code: str) -> tuple[float, int]:
    url = (
        f"{MLB_API_BASE}/people/{person_id}/stats"
        f"?stats=statSplits&group=hitting&season={year}"
        f"&sitCodes={sit_code}&sportId=1"
    )
    try:
        data = _get_json(url)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return np.nan, 0
        stat = splits[0].get("stat", {})
        return _parse_float(stat.get("ops")), int(stat.get("plateAppearances", 0) or 0)
    except Exception:
        return np.nan, 0


def _fetch_bulk_player_split_year(year: int, sit_code: str) -> dict[int, tuple[float, int]]:
    url = (
        f"{MLB_API_BASE}/stats?stats=statSplits&group=hitting&season={year}"
        f"&sitCodes={sit_code}&sportIds=1&playerPool=ALL&limit=10000"
    )
    try:
        data = _get_json(url, timeout=60)
    except Exception as exc:
        print(f"  Bulk split fetch failed for {year} {sit_code}: {exc}", flush=True)
        return {}

    out: dict[int, tuple[float, int]] = {}
    splits = data.get("stats", [{}])[0].get("splits", [])
    for split in splits:
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


def prime_player_split_cache(years: list[int], cache: dict) -> int:
    """
    Bulk-fetch hitter split OPS for complete seasons.
    This turns thousands of player-specific calls into two requests per year.
    """
    updated = 0
    for year in sorted(set(int(y) for y in years)):
        print(f"  Bulk fetching hitter splits {year}...", flush=True)
        vl = _fetch_bulk_player_split_year(year, "vl")
        vr = _fetch_bulk_player_split_year(year, "vr")
        player_ids = set(vl) | set(vr)
        for pid in player_ids:
            l_ops, l_pa = vl.get(pid, (np.nan, 0))
            r_ops, r_pa = vr.get(pid, (np.nan, 0))
            key = f"{pid}:{year}"
            before = cache.get(key)
            cache[key] = {
                "ops_vs_lhp": None if pd.isna(l_ops) else float(l_ops),
                "ops_vs_rhp": None if pd.isna(r_ops) else float(r_ops),
                "pa_vs_lhp": int(l_pa),
                "pa_vs_rhp": int(r_pa),
            }
            updated += before != cache[key]
        print(f"    {year}: {len(player_ids):,} player split rows", flush=True)
    return updated


def get_player_splits(person_id: int,
                      year: int,
                      cache: dict,
                      pause: float = 0.03,
                      no_api_fallback: bool = False) -> dict:
    """Return prior-season split OPS for a player-year."""
    key = f"{person_id}:{year}"
    if key in cache:
        return cache[key]
    if no_api_fallback:
        cache[key] = {
            "ops_vs_lhp": None,
            "ops_vs_rhp": None,
            "pa_vs_lhp": 0,
            "pa_vs_rhp": 0,
        }
        return cache[key]

    ops_l, pa_l = _fetch_player_split(person_id, year, "vl")
    time.sleep(pause)
    ops_r, pa_r = _fetch_player_split(person_id, year, "vr")
    time.sleep(pause)
    result = {
        "ops_vs_lhp": None if pd.isna(ops_l) else float(ops_l),
        "ops_vs_rhp": None if pd.isna(ops_r) else float(ops_r),
        "pa_vs_lhp": int(pa_l),
        "pa_vs_rhp": int(pa_r),
    }
    cache[key] = result
    return result


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


def _lineup_split_features(player_ids: list[int],
                           game_year: int,
                           cache: dict,
                           bulk_primed_years: set[int] | None = None) -> dict:
    split_year = game_year - 1
    bulk_primed_years = bulk_primed_years or set()
    ops_l, ops_r, pa_l, pa_r = [], [], [], []
    for pid in player_ids:
        s = get_player_splits(
            pid,
            split_year,
            cache,
            no_api_fallback=split_year in bulk_primed_years,
        )
        ops_l.append(s.get("ops_vs_lhp"))
        ops_r.append(s.get("ops_vs_rhp"))
        pa_l.append(s.get("pa_vs_lhp", 0))
        pa_r.append(s.get("pa_vs_rhp", 0))
    return {
        "lineup_ops_vs_lhp": _weighted_mean(ops_l, pa_l),
        "lineup_ops_vs_rhp": _weighted_mean(ops_r, pa_r),
        "lineup_known_batters": len(player_ids),
    }


def _write_combined(existing: pd.DataFrame | None,
                    rows: list[dict],
                    out_path: str) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        if existing is not None and not existing.empty else new_df
    )
    if combined.empty:
        return combined
    combined = (
        combined.sort_values(["game_date", "game_pk"])
        .drop_duplicates(["game_pk"], keep="last")
        .reset_index(drop=True)
    )
    combined.to_csv(out_path, index=False)
    return combined


def fetch_historical_lineups(start_date: str,
                             end_date: str,
                             out_path: str = OUT_PATH,
                             refresh: bool = False,
                             flush_every: int = 25,
                             bulk_splits: bool = True) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = pd.read_csv(out_path) if os.path.exists(out_path) and not refresh else None
    existing_keys = set()
    if existing is not None and not existing.empty:
        existing_keys = set(existing["game_pk"].astype(int))

    player_cache = _load_json(PLAYER_SPLIT_CACHE_PATH)
    games = _schedule(start_date, end_date)
    rows = []
    print(f"Schedule games: {len(games):,}; already cached: {len(existing_keys):,}", flush=True)
    if bulk_splits and games:
        split_years = sorted({int(g["game_date"][:4]) - 1 for g in games})
        updated = prime_player_split_cache(split_years, player_cache)
        bulk_primed_years = set(split_years)
        _save_json(PLAYER_SPLIT_CACHE_PATH, player_cache)
        print(
            f"  Bulk cache primed: {updated:,} updated entries; "
            f"{len(player_cache):,} total player-seasons",
            flush=True,
        )
    else:
        bulk_primed_years = set()

    for i, game in enumerate(games, start=1):
        if game["game_pk"] in existing_keys:
            continue
        try:
            lineup = _boxscore_lineup_ids(game["game_pk"])
            year = int(game["game_date"][:4])
            home_feats = _lineup_split_features(
                lineup["home_lineup_ids"], year, player_cache, bulk_primed_years
            )
            away_feats = _lineup_split_features(
                lineup["away_lineup_ids"], year, player_cache, bulk_primed_years
            )
            rows.append({
                **game,
                "home_lineup_ids": " ".join(map(str, lineup["home_lineup_ids"])),
                "away_lineup_ids": " ".join(map(str, lineup["away_lineup_ids"])),
                "home_lineup_names": "; ".join(lineup["home_lineup_names"]),
                "away_lineup_names": "; ".join(lineup["away_lineup_names"]),
                "home_lineup_ops_vs_lhp": home_feats["lineup_ops_vs_lhp"],
                "home_lineup_ops_vs_rhp": home_feats["lineup_ops_vs_rhp"],
                "away_lineup_ops_vs_lhp": away_feats["lineup_ops_vs_lhp"],
                "away_lineup_ops_vs_rhp": away_feats["lineup_ops_vs_rhp"],
                "home_lineup_known_batters": home_feats["lineup_known_batters"],
                "away_lineup_known_batters": away_feats["lineup_known_batters"],
            })
            existing_keys.add(game["game_pk"])
        except Exception as exc:
            print(f"  {game['game_date']} {game['away_team']} @ {game['home_team']}: {exc}", flush=True)
        if rows and len(rows) % flush_every == 0:
            existing = _write_combined(existing, rows, out_path)
            rows = []
            _save_json(PLAYER_SPLIT_CACHE_PATH, player_cache)
            print(
                f"  Saved batch at schedule {i:,}/{len(games):,}; "
                f"total rows {len(existing):,}; player cache {len(player_cache):,}",
                flush=True,
            )
        elif i % 50 == 0:
            _save_json(PLAYER_SPLIT_CACHE_PATH, player_cache)
            print(
                f"  Processed {i:,}/{len(games):,}; pending rows {len(rows):,}; "
                f"player cache {len(player_cache):,}",
                flush=True,
            )
        time.sleep(0.04)

    combined = _write_combined(existing, rows, out_path)
    _save_json(PLAYER_SPLIT_CACHE_PATH, player_cache)
    print(f"Saved {len(combined):,} rows -> {out_path}", flush=True)
    print(f"Player split cache: {len(player_cache):,} player-seasons", flush=True)
    return combined


def _year_dates(start_year: int, end_year: int) -> tuple[str, str]:
    return f"{start_year}-01-01", f"{end_year}-12-31"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical MLB starting lineup split features")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--no-bulk-splits", action="store_true")
    args = parser.parse_args()

    if args.start_date or args.end_date:
        start_date = args.start_date or f"{args.start_year}-01-01"
        end_date = args.end_date or start_date
    else:
        start_date, end_date = _year_dates(args.start_year, args.end_year)

    fetch_historical_lineups(
        start_date,
        end_date,
        refresh=args.refresh,
        flush_every=max(1, args.flush_every),
        bulk_splits=not args.no_bulk_splits,
    )


if __name__ == "__main__":
    main()
