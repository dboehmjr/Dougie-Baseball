"""
Fetch 2026 (current-season) pitcher game logs from the MLB Stats API.

Produces two files that parallel the Retrosheet-sourced historical files:
  data/pitcher_game_logs_live.csv   — per-appearance stats (outs, runs)
  data/game_sp_live.csv             — starting pitcher assignments per game

Pitcher IDs are stored as "mlb_{playerId}" to distinguish them from
Retrosheet IDs and to allow a clean union in feature_engineering.py.

Usage:
    python fetch_pitcher_game_logs_live.py            # fetch all 2026 games
    python fetch_pitcher_game_logs_live.py --year 2026
    python fetch_pitcher_game_logs_live.py --from 2026-04-01
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
LOGS_PATH   = os.path.join(DATA_DIR, "pitcher_game_logs_live.csv")
SP_PATH     = os.path.join(DATA_DIR, "game_sp_live.csv")
API_BASE    = "https://statsapi.mlb.com/api/v1"


def _sync_logs_to_db(logs_df: pd.DataFrame, sp_df: pd.DataFrame) -> None:
    try:
        import database as db_mod
        conn = db_mod.get_connection()
        if not logs_df.empty:
            out = logs_df.copy()
            out["game_date"] = pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
            db_mod.upsert_df(out, "pitcher_game_logs", conn)
        if not sp_df.empty:
            out = sp_df.copy()
            out = out.rename(columns={"Date": "game_date"})
            out["game_date"] = pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
            keep = ["game_date", "year", "home_team", "away_team", "game_number", "game_pk",
                    "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name"]
            db_mod.upsert_df(out[[c for c in keep if c in out.columns]], "game_starters", conn)
        conn.close()
    except Exception as exc:
        print(f"  [pitcher_logs→DB] warning: {exc}")

SEASON_START = date(2026, 3, 18)

# MLB Stats API team ID → our abbreviation
TEAM_IDS = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS",
    112: "CHC", 145: "CHW", 113: "CIN", 114: "CLE",
    115: "COL", 116: "DET", 117: "HOU", 118: "KCR",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL",
    142: "MIN", 121: "NYM", 147: "NYY", 133: "ATH",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA",
    137: "SFG", 138: "STL", 139: "TBR", 140: "TEX",
    141: "TOR", 120: "WSN",
}


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}


def _abbrev(team_id: int) -> str:
    return TEAM_IDS.get(team_id, f"UNK{team_id}")


def _pitcher_id(mlb_id: int) -> str:
    return f"mlb_{mlb_id}"


def fetch_completed_games(start: date, end: date) -> list[dict]:
    """
    Return list of completed game dicts: {gamePk, game_date, home_team, away_team}.
    Chunks into 30-day windows to avoid oversized API responses.
    """
    games = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=29), end)
        data = _get(f"{API_BASE}/schedule", {
            "sportId": 1,
            "startDate": str(chunk_start),
            "endDate":   str(chunk_end),
            "hydrate":   "team",
        })
        for day in data.get("dates", []):
            for g in day.get("games", []):
                if "Final" not in g.get("status", {}).get("detailedState", ""):
                    continue
                home_id = g["teams"]["home"]["team"]["id"]
                away_id = g["teams"]["away"]["team"]["id"]
                games.append({
                    "gamePk":     g["gamePk"],
                    "game_date":  day["date"],
                    "home_team":  _abbrev(home_id),
                    "away_team":  _abbrev(away_id),
                })
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.1)
    return games


def fetch_boxscore(game_pk: int) -> dict:
    return _get(f"{API_BASE}/game/{game_pk}/boxscore")


def parse_boxscore(game: dict, boxscore: dict) -> tuple[list[dict], dict | None]:
    """
    Returns (log_rows, sp_row).
    log_rows: one row per pitcher appearance
    sp_row:   game_sp-style row with home/away SP assignments
    """
    log_rows = []
    sp_row   = None

    home_sp_id = home_sp_name = away_sp_id = away_sp_name = None

    for side_key, side_label, team_side in [("home", "home", 0), ("away", "away", 1)]:
        side = boxscore.get("teams", {}).get(side_key, {})
        pitcher_ids = side.get("pitchers", [])
        players     = side.get("players", {})
        if not pitcher_ids:
            continue

        first_pitcher_id = pitcher_ids[0]

        for i, pid in enumerate(pitcher_ids):
            player = players.get(f"ID{pid}", {})
            name   = player.get("person", {}).get("fullName", "Unknown")
            stats  = player.get("stats", {}).get("pitching", {})

            outs = stats.get("outs")
            runs = stats.get("runs")
            if outs is None or runs is None:
                continue

            is_starter = (pid == first_pitcher_id)
            row = {
                "game_date":     game["game_date"],
                "game_id":       f"mlb_{game['gamePk']}",
                "home_team":     game["home_team"],
                "away_team":     game["away_team"],
                "pitcher_id":    _pitcher_id(pid),
                "pitcher_name":  name,
                "team_side":     team_side,
                "is_starter":    is_starter,
                "outs_recorded": int(outs),
                "runs_allowed":  int(runs),
            }
            log_rows.append(row)

            if is_starter:
                if side_label == "home":
                    home_sp_id   = _pitcher_id(pid)
                    home_sp_name = name
                else:
                    away_sp_id   = _pitcher_id(pid)
                    away_sp_name = name

    if home_sp_id and away_sp_id:
        game_date = pd.to_datetime(game["game_date"])
        sp_row = {
            "Date":          str(game_date.date()),
            "game_pk":       game["gamePk"],
            "game_number":   game.get("game_number", 1),
            "year":          int(game_date.year),
            "home_team":     game["home_team"],
            "away_team":     game["away_team"],
            "home_sp_id":    home_sp_id,
            "home_sp_name":  home_sp_name,
            "away_sp_id":    away_sp_id,
            "away_sp_name":  away_sp_name,
        }

    return log_rows, sp_row


def load_existing(path: str, key_cols: list[str], table: str | None = None) -> tuple[pd.DataFrame, set]:
    df = pd.DataFrame()
    if table:
        try:
            import database as db_mod
            df = db_mod.read_table(table)
            if table == "game_starters" and not df.empty:
                df = df.rename(columns={"game_date": "Date"})
                df["Date"] = pd.to_datetime(df["Date"])
                df = df[df["Date"].dt.date >= SEASON_START].copy()
            if table == "pitcher_game_logs" and not df.empty:
                df["game_date"] = pd.to_datetime(df["game_date"])
                df = df[df["game_date"].dt.date >= SEASON_START].copy()
        except Exception as exc:
            print(f"  [{table} DB read] warning: {exc}")
            df = pd.DataFrame()
    if df.empty and os.path.exists(path):
        df = pd.read_csv(path)
    if any(c not in df.columns for c in key_cols):
        return df, set()
    keyed = df.dropna(subset=key_cols)
    already = set(zip(*[keyed[c].astype(str) for c in key_cols]))
    return df, already


def _attach_game_numbers(games: list[dict]) -> list[dict]:
    if not games:
        return games
    from feature_engineering import add_game_identity

    df = pd.DataFrame(games).rename(columns={"game_date": "Date", "gamePk": "game_pk"})
    df["_source_order"] = pd.to_numeric(df["game_pk"], errors="coerce")
    df = add_game_identity(df)
    df = df.rename(columns={"Date": "game_date", "game_pk": "gamePk"})
    return df.drop(columns=["_source_order"], errors="ignore").to_dict("records")


def _prepare_sp_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable game_number values for same-date/same-team doubleheaders."""
    if df.empty:
        return df
    from feature_engineering import add_game_identity

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    if "game_pk" in out.columns:
        out["_source_order"] = pd.to_numeric(out["game_pk"], errors="coerce")
    out = add_game_identity(out)
    return out


def fetch_and_save(start: date, end: date) -> tuple[int, int]:
    """Fetch from start..end, skip already-cached games. Returns (new_log_rows, new_sp_rows)."""
    log_df, log_already = load_existing(LOGS_PATH, ["game_id"], "pitcher_game_logs")
    sp_df,  sp_already  = load_existing(SP_PATH,   ["game_pk"], "game_starters")
    sp_identity_already = set()
    if not sp_df.empty and {"Date", "home_team", "away_team"}.issubset(sp_df.columns):
        game_numbers = (
            sp_df["game_number"] if "game_number" in sp_df.columns
            else pd.Series(1, index=sp_df.index)
        )
        sp_identity_already = set(zip(
            pd.to_datetime(sp_df["Date"]).dt.strftime("%Y-%m-%d"),
            sp_df["home_team"].astype(str),
            sp_df["away_team"].astype(str),
            game_numbers.astype(str),
        ))

    games = _attach_game_numbers(fetch_completed_games(start, end))
    print(f"  Found {len(games)} completed games {start}–{end}")

    new_logs = []
    new_sps  = []
    skipped  = 0

    for g in games:
        game_id  = f"mlb_{g['gamePk']}"
        sp_key   = (str(g["gamePk"]),)
        sp_identity_key = (
            str(pd.to_datetime(g["game_date"]).date()),
            str(g["home_team"]),
            str(g["away_team"]),
            str(g.get("game_number", 1)),
        )

        if (game_id,) in log_already and (sp_key in sp_already or sp_identity_key in sp_identity_already):
            skipped += 1
            continue

        box = fetch_boxscore(g["gamePk"])
        if not box:
            continue

        log_rows, sp_row = parse_boxscore(g, box)
        new_logs.extend(log_rows)
        if sp_row:
            new_sps.append(sp_row)
        time.sleep(0.12)

    if new_logs:
        combined = pd.concat([log_df, pd.DataFrame(new_logs)], ignore_index=True) \
                   if not log_df.empty else pd.DataFrame(new_logs)
        combined["game_date"] = pd.to_datetime(combined["game_date"])
        combined = combined.sort_values(["game_date", "game_id"]).drop_duplicates(
            subset=["game_id", "pitcher_id"]
        )
        combined.to_csv(LOGS_PATH, index=False)

    if new_sps:
        if not sp_df.empty and ("game_pk" not in sp_df.columns or sp_df["game_pk"].isna().all()):
            fetched_dates = set(pd.to_datetime(pd.DataFrame(new_sps)["Date"]).dt.strftime("%Y-%m-%d"))
            sp_df = sp_df[
                ~pd.to_datetime(sp_df["Date"]).dt.strftime("%Y-%m-%d").isin(fetched_dates)
            ]
        combined_sp = pd.concat([sp_df, pd.DataFrame(new_sps)], ignore_index=True) \
                      if not sp_df.empty else pd.DataFrame(new_sps)
        combined_sp = _prepare_sp_cache(combined_sp)
        combined_sp = (combined_sp.sort_values(["Date", "home_team", "away_team", "game_number"])
                                  .drop_duplicates(subset=["game_id"], keep="last"))
        combined_sp.to_csv(SP_PATH, index=False)

    if new_logs or new_sps:
        sp_sync = combined_sp if new_sps else pd.DataFrame()
        _sync_logs_to_db(
            pd.DataFrame(new_logs) if new_logs else pd.DataFrame(),
            sp_sync,
        )

    print(f"  Skipped (cached): {skipped} | New log rows: {len(new_logs)} | New SP rows: {len(new_sps)}")
    return len(new_logs), len(new_sps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch live pitcher game logs from MLB Stats API")
    parser.add_argument("--year", type=int, default=date.today().year,
                        help="Season year (default: current year)")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date YYYY-MM-DD (overrides --year)")
    parser.add_argument("--through", default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    through = date.fromisoformat(args.through) if args.through else date.today() - timedelta(days=1)
    start   = date.fromisoformat(args.from_date) if args.from_date else SEASON_START

    print(f"\nFetching pitcher game logs {start} → {through}")
    n_logs, n_sp = fetch_and_save(start, through)

    # Summary
    if os.path.exists(LOGS_PATH):
        df = pd.read_csv(LOGS_PATH)
        starters = df[df["is_starter"] == True]
        print(f"\n  pitcher_game_logs_live.csv : {len(df):,} rows  ({starters['game_date'].min()} – {starters['game_date'].max()})")
    if os.path.exists(SP_PATH):
        sp = pd.read_csv(SP_PATH)
        print(f"  game_sp_live.csv           : {len(sp):,} rows")
