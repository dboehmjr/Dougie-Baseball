"""Build Retrosheet run-expectancy state-impact features.

This is the lightweight version of the paper's game-state-delta idea.  It parses
Retrosheet play-by-play events, estimates base/out run expectancy, then writes
pregame rolling RE24-style form features to both CSV and SQLite.

Outputs:
  data/state_impact_features.csv
  data/state_impact_plate_appearances.csv
  SQLite tables: state_impact_features, state_impact_plate_appearances
"""

from __future__ import annotations

import csv
import io
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import database as db
from feature_engineering import normalize_schedule_team
from parse_retrosheet_events import RETRO_TO_ABBREV, count_outs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EVENT_CACHE_DIR = os.path.join(DATA_DIR, "retrosheet_events")
PA_PATH = os.path.join(DATA_DIR, "state_impact_plate_appearances.csv")
FEATURE_PATH = os.path.join(DATA_DIR, "state_impact_features.csv")


def _bases_mask(bases: dict[int, str]) -> int:
    return int(1 in bases) + int(2 in bases) * 2 + int(3 in bases) * 4


def _state_key(outs: int, bases: dict[int, str]) -> str:
    return f"{min(int(outs), 3)}_{_bases_mask(bases)}"


def _parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line]))


def _main_event(event: str) -> str:
    primary = event.split(".")[0]
    return primary.split("/")[0].strip()


def _advance_runners(bases: dict[int, str], steps: int, batter_id: str | None = None) -> tuple[dict[int, str], int]:
    new_bases: dict[int, str] = {}
    runs = 0
    for base in (3, 2, 1):
        runner = bases.get(base)
        if not runner:
            continue
        dest = base + steps
        if dest >= 4:
            runs += 1
        else:
            new_bases[dest] = runner
    if batter_id:
        if steps >= 4:
            runs += 1
        else:
            new_bases[steps] = batter_id
    return new_bases, runs


def _force_walk(bases: dict[int, str], batter_id: str) -> tuple[dict[int, str], int]:
    new_bases = dict(bases)
    runs = 0
    if 1 in new_bases and 2 in new_bases and 3 in new_bases:
        runs += 1
    if 1 in new_bases and 2 in new_bases:
        new_bases[3] = new_bases[2]
    if 1 in new_bases:
        new_bases[2] = new_bases[1]
    new_bases[1] = batter_id
    return new_bases, runs


def _apply_default_movement(event: str, bases: dict[int, str], batter_id: str) -> tuple[dict[int, str], int]:
    ev = _main_event(event)
    if ev.startswith("HR"):
        new_bases, runs = _advance_runners(bases, 4, batter_id)
        return new_bases, runs
    if ev.startswith("T"):
        return _advance_runners(bases, 3, batter_id)
    if ev.startswith("D"):
        return _advance_runners(bases, 2, batter_id)
    if ev.startswith("S"):
        return _advance_runners(bases, 1, batter_id)
    if ev.startswith(("W", "IW", "HP")):
        return _force_walk(bases, batter_id)
    if ev.startswith(("E", "FC")):
        return _advance_runners(bases, 1, batter_id)
    return dict(bases), 0


def _remove_runner(bases: dict[int, str], runner_id: str) -> None:
    for base, runner in list(bases.items()):
        if runner == runner_id:
            bases.pop(base, None)


def _apply_explicit_advances(event: str, bases: dict[int, str], batter_id: str,
                             starting_bases: dict[int, str]) -> tuple[dict[int, str], int]:
    if "." not in event:
        return bases, 0
    new_bases = dict(bases)
    runs = 0
    for adv in event.split(".", 1)[1].split(";"):
        adv = adv.strip()
        m = re.match(r"^([B123])([-X])([123H])", adv)
        if not m:
            continue
        src, marker, dest = m.groups()
        runner = batter_id if src == "B" else starting_bases.get(int(src))
        if runner:
            _remove_runner(new_bases, runner)
        if marker == "X" or not runner:
            continue
        if dest == "H":
            runs += 1
        else:
            new_bases[int(dest)] = runner
    return new_bases, runs


def _apply_play(event: str, bases: dict[int, str], outs: int, batter_id: str) -> tuple[dict[int, str], int, int]:
    default_bases, default_runs = _apply_default_movement(event, bases, batter_id)
    if "." in event:
        explicit_bases, explicit_runs = _apply_explicit_advances(event, default_bases, batter_id, bases)
        # Start from normal event movement, then let explicit runner advances
        # override individual runners. Retrosheet often omits B-1 on singles.
        next_bases, runs = explicit_bases, explicit_runs
    else:
        next_bases, runs = default_bases, default_runs
    next_outs = min(3, outs + count_outs(event))
    if next_outs >= 3:
        next_bases = {}
    return next_bases, next_outs, runs


def parse_event_file(lines: list[str]) -> list[dict]:
    records: list[dict] = []
    game_id = game_date = vis_team = home_team = None
    pitcher: dict[int, str] = {}
    pitcher_meta: dict[tuple[int, str], tuple[str, bool]] = {}
    inning = batting_side = None
    outs = 0
    bases: dict[int, str] = {}
    half_seq = -1
    runs_before = 0

    def reset_half(new_inning, new_batting_side):
        nonlocal inning, batting_side, outs, bases, half_seq, runs_before
        inning = int(new_inning)
        batting_side = int(new_batting_side)
        outs = 0
        bases = {}
        half_seq += 1
        runs_before = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        fields = _parse_csv_line(line)
        rec = fields[0]
        if rec == "id":
            game_id = fields[1] if len(fields) > 1 else None
            if game_id and len(game_id) >= 11:
                game_date = pd.Timestamp(
                    year=int(game_id[3:7]), month=int(game_id[7:9]), day=int(game_id[9:11])
                )
            vis_team = home_team = None
            pitcher = {}
            pitcher_meta = {}
            inning = batting_side = None
            outs = 0
            bases = {}
            half_seq = -1
            runs_before = 0
        elif rec == "info" and len(fields) >= 3:
            if fields[1] == "visteam":
                vis_team = fields[2]
            elif fields[1] == "hometeam":
                home_team = fields[2]
        elif rec in ("start", "sub") and len(fields) >= 6:
            side = int(fields[3])
            pos = int(fields[5])
            if pos == 1:
                pid = fields[1]
                pitcher[side] = pid
                pitcher_meta.setdefault((side, pid), (fields[2], rec == "start"))
        elif rec == "play" and len(fields) >= 7 and game_id:
            play_inning = int(fields[1])
            play_batting_side = int(fields[2])
            if inning != play_inning or batting_side != play_batting_side:
                reset_half(play_inning, play_batting_side)
            pitching_side = 1 - play_batting_side
            pitcher_id = pitcher.get(pitching_side)
            event = fields[6]
            before_outs = outs
            before_bases = dict(bases)
            after_bases, after_outs, runs = _apply_play(event, before_bases, before_outs, fields[3])

            ht = RETRO_TO_ABBREV.get(home_team, home_team)
            at = RETRO_TO_ABBREV.get(vis_team, vis_team)
            batting_team = at if play_batting_side == 0 else ht
            pitching_team = ht if play_batting_side == 0 else at
            pname, is_starter = pitcher_meta.get((pitching_side, pitcher_id), ("unknown", False))
            records.append({
                "game_id": game_id,
                "game_date": game_date,
                "home_team": normalize_schedule_team(ht),
                "away_team": normalize_schedule_team(at),
                "half_id": f"{game_id}_{half_seq}",
                "inning": play_inning,
                "batting_side": play_batting_side,
                "batting_team": normalize_schedule_team(batting_team),
                "pitching_team": normalize_schedule_team(pitching_team),
                "pitcher_id": pitcher_id,
                "pitcher_name": pname,
                "pitcher_is_starter": bool(is_starter),
                "state_before": _state_key(before_outs, before_bases),
                "state_after": _state_key(after_outs, after_bases),
                "outs_before": before_outs,
                "outs_after": after_outs,
                "runs_scored": runs,
                "runs_before_half": runs_before,
            })
            bases, outs = after_bases, after_outs
            runs_before += runs
    return records


def download_event_zip(year: int) -> bytes:
    os.makedirs(EVENT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(EVENT_CACHE_DIR, f"{year}eve.zip")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    url = f"https://www.retrosheet.org/events/{year}eve.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(r.content)
    time.sleep(1.0)
    return r.content


def build_plate_appearances(start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        print(f"Parsing Retrosheet events for {year}...")
        try:
            content = download_event_zip(year)
        except Exception as exc:
            print(f"  skipped {year}: {exc}")
            continue
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            ev_files = [n for n in zf.namelist() if n.endswith((".EVA", ".EVN"))]
            for ev_name in ev_files:
                with zf.open(ev_name) as f:
                    lines = [line.decode("latin-1") for line in f]
                rows.extend(parse_event_file(lines))
        print(f"  cumulative play records: {len(rows):,}")
    pa = pd.DataFrame(rows)
    if pa.empty:
        return pa
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    half_total = pa.groupby("half_id")["runs_scored"].sum().rename("half_total_runs")
    pa = pa.merge(half_total, on="half_id", how="left")
    pa["runs_to_end_before"] = pa["half_total_runs"] - pa["runs_before_half"]
    re_table = pa.groupby("state_before")["runs_to_end_before"].mean().to_dict()
    pa["re_before"] = pa["state_before"].map(re_table).fillna(0.0)
    pa["re_after"] = np.where(
        pa["outs_after"] >= 3,
        0.0,
        pa["state_after"].map(re_table).fillna(0.0),
    )
    pa["offense_re24"] = pa["runs_scored"] + pa["re_after"] - pa["re_before"]
    pa["pitching_re24"] = -pa["offense_re24"]
    pa = pa.sort_values(["game_date", "home_team", "away_team", "game_id"]).reset_index(drop=True)
    pa["game_number"] = (
        pa[["game_date", "home_team", "away_team", "game_id"]]
        .drop_duplicates()
        .groupby(["game_date", "home_team", "away_team"], dropna=False)
        .cumcount()
        .add(1)
        .reindex(pa.drop_duplicates(["game_date", "home_team", "away_team", "game_id"]).index)
    )
    game_numbers = (
        pa[["game_date", "home_team", "away_team", "game_id"]]
        .drop_duplicates()
        .sort_values(["game_date", "home_team", "away_team", "game_id"])
    )
    game_numbers["game_number"] = game_numbers.groupby(
        ["game_date", "home_team", "away_team"], dropna=False
    ).cumcount().add(1)
    pa = pa.drop(columns=["game_number"], errors="ignore").merge(
        game_numbers, on=["game_date", "home_team", "away_team", "game_id"], how="left"
    )
    return pa


def build_feature_table(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    starter_mask = pa["pitcher_is_starter"].fillna(False).astype(bool)
    games = (
        pa[["game_date", "home_team", "away_team", "game_number", "game_id"]]
        .drop_duplicates()
        .sort_values(["game_date", "home_team", "away_team", "game_number"])
    )

    offense_game = (
        pa.groupby(["game_date", "game_id", "batting_team"], dropna=False)
        .agg(offense_re24=("offense_re24", "sum"), offense_pa=("offense_re24", "size"))
        .reset_index()
        .sort_values(["batting_team", "game_date", "game_id"])
    )
    offense_game["team_offense_re24_15g"] = offense_game.groupby("batting_team")["offense_re24"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=5).mean()
    )

    starter_game = (
        pa[starter_mask]
        .groupby(["game_date", "game_id", "pitching_team", "pitcher_id"], dropna=False)
        .agg(sp_re24=("pitching_re24", "sum"))
        .reset_index()
        .sort_values(["pitcher_id", "game_date", "game_id"])
    )
    starter_game["sp_re24_last3"] = starter_game.groupby("pitcher_id")["sp_re24"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )

    bullpen_game = (
        pa[~starter_mask]
        .groupby(["pitching_team", "game_date"], dropna=False)
        .agg(bullpen_re24=("pitching_re24", "sum"))
        .reset_index()
        .sort_values(["pitching_team", "game_date"])
    )
    bullpen_parts = []
    for _, grp in bullpen_game.groupby("pitching_team"):
        work = grp.set_index("game_date").sort_index()
        work["bullpen_re24_15d"] = work["bullpen_re24"].rolling("15D", closed="left").mean()
        bullpen_parts.append(work.reset_index())
    bullpen_game = pd.concat(bullpen_parts, ignore_index=True) if bullpen_parts else bullpen_game

    out = games.copy()
    out = out.merge(
        offense_game.rename(columns={
            "batting_team": "home_team",
            "team_offense_re24_15g": "home_offense_re24_15g",
        })[["game_date", "game_id", "home_team", "home_offense_re24_15g"]],
        on=["game_date", "game_id", "home_team"], how="left",
    )
    out = out.merge(
        offense_game.rename(columns={
            "batting_team": "away_team",
            "team_offense_re24_15g": "away_offense_re24_15g",
        })[["game_date", "game_id", "away_team", "away_offense_re24_15g"]],
        on=["game_date", "game_id", "away_team"], how="left",
    )
    out = out.merge(
        starter_game.rename(columns={
            "pitching_team": "home_team",
            "sp_re24_last3": "home_sp_re24_last3",
        })[["game_date", "game_id", "home_team", "home_sp_re24_last3"]],
        on=["game_date", "game_id", "home_team"], how="left",
    )
    out = out.merge(
        starter_game.rename(columns={
            "pitching_team": "away_team",
            "sp_re24_last3": "away_sp_re24_last3",
        })[["game_date", "game_id", "away_team", "away_sp_re24_last3"]],
        on=["game_date", "game_id", "away_team"], how="left",
    )
    out = out.merge(
        bullpen_game.rename(columns={
            "pitching_team": "home_team",
            "bullpen_re24_15d": "home_bullpen_re24_15d",
        })[["game_date", "home_team", "home_bullpen_re24_15d"]],
        on=["game_date", "home_team"], how="left",
    )
    out = out.merge(
        bullpen_game.rename(columns={
            "pitching_team": "away_team",
            "bullpen_re24_15d": "away_bullpen_re24_15d",
        })[["game_date", "away_team", "away_bullpen_re24_15d"]],
        on=["game_date", "away_team"], how="left",
    )
    out["offense_re24_diff"] = out["home_offense_re24_15g"] - out["away_offense_re24_15g"]
    out["sp_re24_diff"] = out["home_sp_re24_last3"] - out["away_sp_re24_last3"]
    out["bullpen_re24_diff"] = out["home_bullpen_re24_15d"] - out["away_bullpen_re24_15d"]
    return out


def main(start_year: int = 2015, end_year: int = 2025) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    pa = build_plate_appearances(start_year, end_year)
    if pa.empty:
        raise RuntimeError("No Retrosheet play records parsed.")
    features = build_feature_table(pa)

    pa_out = pa.copy()
    pa_out["game_date"] = pa_out["game_date"].dt.strftime("%Y-%m-%d")
    feat_out = features.copy()
    feat_out["game_date"] = pd.to_datetime(feat_out["game_date"]).dt.strftime("%Y-%m-%d")

    pa_out.to_csv(PA_PATH, index=False)
    feat_out.to_csv(FEATURE_PATH, index=False)

    conn = db.get_connection()
    db.replace_table(pa_out, "state_impact_plate_appearances", conn)
    db.replace_table(feat_out, "state_impact_features", conn)
    conn.close()

    print(f"Saved {len(pa_out):,} play records -> {PA_PATH}")
    print(f"Saved {len(feat_out):,} game feature rows -> {FEATURE_PATH}")
    print("Updated SQLite tables: state_impact_plate_appearances, state_impact_features")


if __name__ == "__main__":
    main()
