"""Fetch live MLBAM play-by-play and update state-impact features.

This extends build_state_impact_features.py beyond Retrosheet availability by
pulling completed current-season games from the MLB Stats API, converting each
at-bat to the same RE24-style plate appearance table, then rebuilding the
state_impact_features table used by feature_engineering.py.

Usage:
  python fetch_state_impact_live.py
  python fetch_state_impact_live.py --from 2026-03-18 --through 2026-05-11
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

import database as db
from build_state_impact_features import (
    FEATURE_PATH,
    PA_PATH,
    build_feature_table,
    _bases_mask,
    _state_key,
)
from feature_engineering import normalize_schedule_team
from fetch_pitcher_game_logs_live import SEASON_START, fetch_completed_games

API_BASE = "https://statsapi.mlb.com/api/v1"


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}


def _mlb_pid(person: dict | None) -> str | None:
    if not person or person.get("id") is None:
        return None
    return f"mlb_{person['id']}"


def _post_bases(matchup: dict, outs_after: int) -> dict[int, str]:
    if outs_after >= 3:
        return {}
    mapping = {
        1: matchup.get("postOnFirst"),
        2: matchup.get("postOnSecond"),
        3: matchup.get("postOnThird"),
    }
    return {base: _mlb_pid(person) for base, person in mapping.items() if _mlb_pid(person)}


def fetch_play_by_play(game_pk: int) -> dict:
    return _get(f"{API_BASE}/game/{game_pk}/playByPlay")


def parse_game(game: dict, play_by_play: dict) -> list[dict]:
    plays = play_by_play.get("allPlays", [])
    if not plays:
        return []

    game_id = f"mlb_{game['gamePk']}"
    game_date = pd.Timestamp(game["game_date"])
    home_team = normalize_schedule_team(game["home_team"])
    away_team = normalize_schedule_team(game["away_team"])
    game_number = int(game.get("game_number", 1))

    outs = 0
    bases: dict[int, str] = {}
    half_id = None
    half_seq = -1
    runs_before_half = 0
    home_score = 0
    away_score = 0
    starter_pitchers: dict[str, str] = {}
    rows: list[dict] = []

    for play in plays:
        about = play.get("about", {})
        if not about.get("isComplete", True):
            continue
        inning = int(about.get("inning", 0) or 0)
        half = str(about.get("halfInning") or "")
        current_half_id = f"{game_id}_{inning}_{half}"
        if current_half_id != half_id:
            half_id = current_half_id
            half_seq += 1
            outs = 0
            bases = {}
            runs_before_half = 0

        is_top = bool(about.get("isTopInning"))
        batting_side = 0 if is_top else 1
        batting_team = away_team if is_top else home_team
        pitching_team = home_team if is_top else away_team

        matchup = play.get("matchup", {})
        result = play.get("result", {})
        pitcher_id = _mlb_pid(matchup.get("pitcher"))
        pitcher_name = (matchup.get("pitcher") or {}).get("fullName", "Unknown")
        if pitcher_id and pitching_team not in starter_pitchers:
            starter_pitchers[pitching_team] = pitcher_id

        state_before = _state_key(outs, bases)
        before_score = away_score if is_top else home_score
        new_away_score = int(result.get("awayScore", away_score) or 0)
        new_home_score = int(result.get("homeScore", home_score) or 0)
        after_score = new_away_score if is_top else new_home_score
        runs_scored = max(0, after_score - before_score)

        count = play.get("count", {})
        outs_after = int(count.get("outs", outs) or 0)
        bases_after = _post_bases(matchup, outs_after)
        state_after = _state_key(outs_after, bases_after)

        rows.append({
            "game_id": game_id,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "game_number": game_number,
            "half_id": f"{game_id}_{half_seq}",
            "inning": inning,
            "batting_side": batting_side,
            "batting_team": batting_team,
            "pitching_team": pitching_team,
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_name,
            "pitcher_is_starter": pitcher_id == starter_pitchers.get(pitching_team),
            "state_before": state_before,
            "state_after": state_after,
            "outs_before": outs,
            "outs_after": outs_after,
            "runs_scored": runs_scored,
            "runs_before_half": runs_before_half,
            "source": "mlb_stats_api",
        })

        outs = outs_after
        bases = bases_after
        away_score = new_away_score
        home_score = new_home_score
        runs_before_half += runs_scored

    return rows


def _load_existing_plate_appearances() -> pd.DataFrame:
    if db.table_exists("state_impact_plate_appearances"):
        df = db.read_table("state_impact_plate_appearances")
    elif os.path.exists(PA_PATH):
        df = pd.read_csv(PA_PATH)
    else:
        df = pd.DataFrame()
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        if "source" not in df.columns:
            df["source"] = np.where(df["game_id"].astype(str).str.startswith("mlb_"), "mlb_stats_api", "retrosheet")
    return df


def _recompute_re24(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["runs_scored"] = pd.to_numeric(pa["runs_scored"], errors="coerce").fillna(0.0)
    pa["runs_before_half"] = pd.to_numeric(pa["runs_before_half"], errors="coerce").fillna(0.0)
    half_total = pa.groupby("half_id")["runs_scored"].sum().rename("half_total_runs")
    pa = pa.drop(columns=["half_total_runs"], errors="ignore").merge(half_total, on="half_id", how="left")
    pa["runs_to_end_before"] = pa["half_total_runs"] - pa["runs_before_half"]
    re_table = pa.groupby("state_before")["runs_to_end_before"].mean().to_dict()
    pa["re_before"] = pa["state_before"].map(re_table).fillna(0.0)
    pa["outs_after"] = pd.to_numeric(pa["outs_after"], errors="coerce").fillna(0).astype(int)
    pa["re_after"] = np.where(
        pa["outs_after"] >= 3,
        0.0,
        pa["state_after"].map(re_table).fillna(0.0),
    )
    pa["offense_re24"] = pa["runs_scored"] + pa["re_after"] - pa["re_before"]
    pa["pitching_re24"] = -pa["offense_re24"]
    if "game_number" not in pa.columns:
        pa["game_number"] = 1
    pa["game_number"] = pd.to_numeric(pa["game_number"], errors="coerce").fillna(1).astype(int)
    return pa


def fetch_live_plate_appearances(start: date, through: date) -> pd.DataFrame:
    games = fetch_completed_games(start, through)
    if not games:
        return pd.DataFrame()
    from fetch_pitcher_game_logs_live import _attach_game_numbers

    games = _attach_game_numbers(games)
    rows: list[dict] = []
    print(f"  Found {len(games)} completed games {start}–{through}")
    for game in games:
        data = fetch_play_by_play(game["gamePk"])
        game_rows = parse_game(game, data)
        rows.extend(game_rows)
        time.sleep(0.08)
    return pd.DataFrame(rows)


def update_live_state_impact(start: date, through: date) -> tuple[int, int]:
    existing = _load_existing_plate_appearances()
    live_pa = fetch_live_plate_appearances(start, through)
    if live_pa.empty:
        print("  No live play-by-play records fetched.")
        return 0, 0

    existing_non_live = existing[
        ~existing["game_id"].astype(str).str.startswith("mlb_")
    ].copy() if not existing.empty else pd.DataFrame()
    existing_live = existing[
        existing["game_id"].astype(str).str.startswith("mlb_")
    ].copy() if not existing.empty else pd.DataFrame()

    fetched_game_ids = set(live_pa["game_id"].astype(str))
    existing_live = existing_live[
        ~existing_live["game_id"].astype(str).isin(fetched_game_ids)
    ]
    combined = pd.concat([existing_non_live, existing_live, live_pa], ignore_index=True, sort=False)
    combined = _recompute_re24(combined)
    features = build_feature_table(combined)

    pa_out = combined.copy()
    pa_out["game_date"] = pd.to_datetime(pa_out["game_date"]).dt.strftime("%Y-%m-%d")
    feat_out = features.copy()
    feat_out["game_date"] = pd.to_datetime(feat_out["game_date"]).dt.strftime("%Y-%m-%d")

    pa_out.to_csv(PA_PATH, index=False)
    feat_out.to_csv(FEATURE_PATH, index=False)

    conn = db.get_connection()
    db.replace_table(pa_out, "state_impact_plate_appearances", conn)
    db.replace_table(feat_out, "state_impact_features", conn)
    conn.close()

    return len(live_pa), len(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch live state-impact features from MLB Stats API")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (default: 2026 season start)")
    parser.add_argument("--through", default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    through = date.fromisoformat(args.through) if args.through else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.from_date) if args.from_date else SEASON_START
    print(f"\nFetching live state-impact play-by-play {start} → {through}")
    n_pa, n_features = update_live_state_impact(start, through)
    print(f"  Live play records fetched: {n_pa:,}")
    print(f"  state_impact_features rows: {n_features:,}")
