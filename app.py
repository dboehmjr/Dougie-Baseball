"""
MLB Win Probability — Streamlit App

Pages:
  1. Today's Slate   — auto-fetch schedule, show predictions + Kelly stakes
  2. Custom Game     — pick any matchup + umpire
  3. Bankroll Tracker — log results, track P&L over time
"""

import os
import re
import sys
import unicodedata
import warnings
import requests
import zoneinfo
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from predict import predict_matchup
from fetch_odds import load_or_fetch_odds, get_home_implied_prob, get_moneyline_str
from fetch_lineups import fetch_confirmed_lineups
from fetch_weather import wind_to_cf, FULL_DOME

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JUICE      = 110
NET_ODDS   = 100 / JUICE
BREAKEVEN  = JUICE / (JUICE + 100)

ALL_TEAMS = [
    "ARI","ATL","BAL","BOS","CHC","CHW","CIN","CLE","COL","DET",
    "HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK",
    "PHI","PIT","SDP","SEA","SFG","STL","TBR","TEX","TOR","WSN",
]

MLB_API_MAP = {
    "WSH": "WSN", "SD": "SDP", "TB": "TBR",
    "KC": "KCR", "AZ": "ARI", "SF": "SFG",
    "ATH": "OAK", "CWS": "CHW",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LEDGER_PATH = os.path.join(DATA_DIR, "bankroll_ledger.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def kelly_stake(p: float, bankroll: float, frac: float = 0.25,
                max_pct: float = 0.10, ml: float = -110) -> float:
    """Kelly stake using actual moneyline odds."""
    net = (ml / 100) if ml > 0 else (100 / abs(ml))
    edge = p * net - (1 - p)
    if edge <= 0:
        return 0.0
    k = frac * (edge / net)
    return round(min(k, max_pct) * bankroll, 2)


def confidence_color(conf: float) -> str:
    if conf >= 0.65:
        return "🟢"
    if conf >= 0.58:
        return "🟡"
    return "⚪"


@st.cache_data(ttl=60)
def fetch_schedule(game_date: str) -> list[dict]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={game_date}&hydrate=team,linescore,probablePitcher")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("dates", [{}])[0].get("games", [])
        out = []
        for g in games:
            home_raw = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "UNK")
            away_raw = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "UNK")
            if home_raw == "UNK" or away_raw == "UNK":
                continue
            home = MLB_API_MAP.get(home_raw, home_raw)
            away = MLB_API_MAP.get(away_raw, away_raw)
            status = g.get("status", {}).get("detailedState", "")
            h_score = g.get("teams", {}).get("home", {}).get("score")
            a_score = g.get("teams", {}).get("away", {}).get("score")
            home_sp = g.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
            away_sp = g.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
            # Game time in Eastern
            try:
                utc_str = g.get("gameDate", "")
                dt_utc  = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                dt_et   = dt_utc.astimezone(zoneinfo.ZoneInfo("America/New_York"))
                game_time = dt_et.strftime("%-I:%M %p ET")
            except Exception:
                game_time = "TBD"

            # Live game state from linescore
            ls          = g.get("linescore", {})
            ls_teams    = ls.get("teams", {})
            inning      = ls.get("currentInning")
            inning_ord  = ls.get("currentInningOrdinal", "")
            inning_state= ls.get("inningState", "")   # Top/Middle/Bottom/End
            outs        = ls.get("outs")
            live_h_runs = ls_teams.get("home", {}).get("runs")
            live_a_runs = ls_teams.get("away", {}).get("runs")

            # Build compact live string
            if "Final" in status:
                live_str = f"Final  {live_a_runs}–{live_h_runs}"
            elif "Progress" in status:
                half = "▲" if ls.get("isTopInning") else "▼"
                live_str = f"{half}{inning_ord}  {live_a_runs}–{live_h_runs}  {outs} out"
            elif inning_state in ("Middle", "End"):
                half = "Mid" if inning_state == "Middle" else "End"
                live_str = f"{half} {inning_ord}  {live_a_runs}–{live_h_runs}"
            elif "Delayed" in status:
                live_str = "⚡ Delayed"
            elif "Pre-Game" in status or "Warmup" in status:
                live_str = f"🔜 {game_time}"
            else:
                live_str = game_time

            out.append({
                "home": home, "away": away,
                "status": status,
                "home_score": h_score,
                "away_score": a_score,
                "home_sp": home_sp,
                "away_sp": away_sp,
                "game_time": game_time,
                "live_str":  live_str,
                "inning":    inning,
                "outs":      outs,
            })
        return out
    except Exception as e:
        st.error(f"Could not fetch schedule: {e}")
        return []


@st.cache_data(ttl=600, show_spinner=False)
def get_prediction(home: str, away: str, year: int, game_date: str,
                   home_sp: str = "", away_sp: str = "") -> float:
    return predict_matchup(home, away, year, game_date=game_date,
                           home_sp_name=home_sp or None,
                           away_sp_name=away_sp or None,
                           verbose=False)


@st.cache_data(ttl=900, show_spinner=False)
def load_odds_for_date(game_date: str) -> pd.DataFrame:
    """Load Vegas moneylines for a game date (cached 15 min)."""
    try:
        return load_or_fetch_odds(game_date=game_date)
    except Exception as e:
        st.warning(f"Could not load Vegas odds for {game_date}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_lineups_for_date(game_date: str) -> pd.DataFrame:
    """Load confirmed lineup status for all games on a date (refreshes every 5 min)."""
    try:
        return fetch_confirmed_lineups(game_date=game_date)
    except Exception as e:
        st.warning(f"Could not load lineup status for {game_date}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_pitcher_stuff() -> pd.DataFrame:
    """Load FanGraphs pitcher stuff CSV."""
    stuff_path = os.path.join(DATA_DIR, "pitcher_stuff.csv")
    if not os.path.exists(stuff_path):
        return pd.DataFrame()
    return pd.read_csv(stuff_path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_weather_data() -> pd.DataFrame:
    weather_path = os.path.join(DATA_DIR, "weather.csv")
    if not os.path.exists(weather_path):
        return pd.DataFrame()
    return pd.read_csv(weather_path)


def _normalize_sp_name(name: str) -> str:
    """Strip accents and normalize to lowercase — matches pitcher_stuff.csv name_norm."""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z ]", "", name.lower().strip())
    return re.sub(r"\s+", " ", name).strip()


def get_sp_stats(name: str, team: str, year: int, stuff_df: pd.DataFrame) -> dict:
    """Look up pitcher stats from stuff_df by normalized name + year."""
    if stuff_df.empty or not name or name == "TBD":
        return {}
    normalized = _normalize_sp_name(name)
    mask = (stuff_df["name_norm"] == normalized) & (stuff_df["year"] == year)
    if not mask.any():
        # Try last-name fallback
        last = normalized.split()[-1]
        mask = (stuff_df["name_norm"].str.contains(last, na=False)) & (stuff_df["year"] == year)
    if not mask.any():
        # Prior year
        mask = stuff_df["name_norm"] == normalized
    if not mask.any():
        return {}
    row = stuff_df[mask].sort_values("year", ascending=False).iloc[0]
    result = {"hand": row.get("Throws", "")}
    for col in ["xFIP", "FBv", "SwStr_pct", "K_pct", "GS", "pa"]:
        val = row.get(col)
        if pd.notna(val):
            result[col] = float(val)
    return result


def get_weather_for_game(home_team: str, game_date: str, weather_df: pd.DataFrame) -> dict:
    if not weather_df.empty:
        mask = (weather_df["home_team"] == home_team) & (weather_df["date"].astype(str).str[:10] == game_date[:10])
        row = weather_df[mask]
        if not row.empty:
            r = row.iloc[0]
            return {
                "temp_f":        float(r["temp_f"]) if pd.notna(r.get("temp_f")) else None,
                "wind_speed_mph": float(r["wind_speed_mph"]) if pd.notna(r.get("wind_speed_mph")) else None,
                "wind_dir_deg":  float(r["wind_dir_deg"]) if pd.notna(r.get("wind_dir_deg")) else None,
                "humidity_pct":  float(r["humidity_pct"]) if pd.notna(r.get("humidity_pct")) else None,
            }
    # Fall back to live forecast API for today/upcoming games not yet in cache
    try:
        from fetch_weather import get_game_weather
        wx = get_game_weather(home_team, game_date, weather_df)
        return wx if wx else {}
    except Exception:
        return {}


def generate_key_factors(r: dict, home_stuff: dict, away_stuff: dict,
                         team_form: dict, weather: dict, year: int) -> list[str]:
    """Return a list of plain-English bullet points explaining the pick."""
    factors = []
    home, away, pick = r["home"], r["away"], r["pick"]

    # Pitching edge
    h_xfip = home_stuff.get("xFIP")
    a_xfip = away_stuff.get("xFIP")
    if h_xfip and a_xfip:
        diff = a_xfip - h_xfip
        if diff >= 0.5:
            factors.append(
                f"Pitching edge: {r['home_sp']} has lower xFIP than {r['away_sp']} "
                f"({h_xfip:.2f} vs {a_xfip:.2f})"
            )
        elif diff <= -0.5:
            factors.append(
                f"Pitching edge: {r['away_sp']} has lower xFIP than {r['home_sp']} "
                f"({a_xfip:.2f} vs {h_xfip:.2f})"
            )

    # Market edge
    edge = r.get("model_edge", 0) or 0
    if abs(edge) >= 0.05:
        dir_ = "above" if edge > 0 else "below"
        factors.append(
            f"Market edge: model is {abs(edge):.1%} {dir_} Vegas on {pick}"
        )

    # Team momentum / streaks
    for team, label in [(home, "home"), (away, "away")]:
        if team not in team_form:
            continue
        f = team_form[team]
        streak  = f.get("streak", 0)
        mom     = f.get("momentum", np.nan)
        if streak >= 4:
            factors.append(f"{team} on a {streak}-game win streak")
        elif streak <= -4:
            factors.append(f"{team} losers in {abs(streak)} straight")
        if not pd.isna(mom):
            if mom >= 0.7:
                factors.append(f"{team} offense trending up (run diff +{mom:.1f} over last 7 vs 15 games)")
            elif mom <= -0.7:
                factors.append(f"{team} offense cooling off (run diff {mom:.1f} over last 7 vs 15 games)")

    # Home underdog value
    if pick == home and r.get("vegas_prob", 0.5) < 0.5 and r["home%"] > 0.52:
        factors.append(f"Home underdog value: model gives {home} {r['home%']:.1%} despite Vegas having them as dog")

    # Weather
    if weather and home not in FULL_DOME:
        spd = weather.get("wind_speed_mph")
        deg = weather.get("wind_dir_deg")
        temp = weather.get("temp_f")
        if spd and deg and spd >= 8:
            wtcf = wind_to_cf(spd, deg, home)
            if wtcf >= 7:
                factors.append(f"Wind blowing out at {spd:.0f} mph — hitter-friendly conditions")
            elif wtcf <= -7:
                factors.append(f"Wind blowing in at {spd:.0f} mph — pitcher-friendly conditions")
        if temp and temp >= 88:
            factors.append(f"Hot weather ({temp:.0f}°F) slightly favors hitters")
        elif temp and temp <= 45:
            factors.append(f"Cold weather ({temp:.0f}°F) slightly suppresses offense")

    if not factors:
        factors.append(f"Model gives {pick} a {r['conf']:.1%} win probability vs Vegas implied {r.get('vegas_prob', 0.5):.1%}")

    return factors


LEDGER_COLS = ["date","matchup","bet_side","stake","odds","won","pnl","bankroll","notes"]
STARTING_BANKROLL = 100.0

PRED_LOG_PATH = os.path.join(DATA_DIR, "predictions_log.csv")
PRED_LOG_COLS = [
    "date","home","away","home_sp","away_sp",
    "model_home_prob","vegas_home_prob","pick",
    "home_score","away_score","home_win","model_correct",
]


def load_pred_log() -> pd.DataFrame:
    if os.path.exists(PRED_LOG_PATH):
        return pd.read_csv(PRED_LOG_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=PRED_LOG_COLS)


def save_pred_log(df: pd.DataFrame) -> None:
    df.to_csv(PRED_LOG_PATH, index=False)


def log_or_update_predictions(rows: list, game_date: str) -> int:
    """
    Write predictions to predictions_log.csv for every game on the slate.
    Adds new rows for games not yet logged; fills in scores when games go Final.
    Returns number of new rows added.
    """
    log = load_pred_log()
    date_ts = pd.Timestamp(game_date)

    existing_keys = (
        set(zip(log["date"].astype(str).str[:10], log["home"], log["away"]))
        if not log.empty else set()
    )

    new_rows, updates = [], 0
    for r in rows:
        home, away = r["home"], r["away"]
        key = (game_date, home, away)
        home_win_val = r.get("home_win")
        model_correct = None
        if home_win_val is not None:
            model_correct = bool((r["home%"] > 0.5) == bool(home_win_val))

        if key not in existing_keys:
            new_rows.append({
                "date":            date_ts,
                "home":            home,
                "away":            away,
                "home_sp":         r.get("home_sp", ""),
                "away_sp":         r.get("away_sp", ""),
                "model_home_prob": round(r["home%"], 4),
                "vegas_home_prob": r.get("vegas_prob", np.nan),
                "pick":            r["pick"],
                "home_score":      r.get("home_score"),
                "away_score":      r.get("away_score"),
                "home_win":        home_win_val,
                "model_correct":   model_correct,
            })
        elif home_win_val is not None and not log.empty:
            # Fill in scores for a previously logged game that just went Final
            mask = (
                (log["date"].astype(str).str[:10] == game_date)
                & (log["home"] == home)
                & (log["away"] == away)
                & log["home_win"].isna()
            )
            if mask.any():
                log.loc[mask, "home_score"]    = r.get("home_score")
                log.loc[mask, "away_score"]    = r.get("away_score")
                log.loc[mask, "home_win"]      = home_win_val
                log.loc[mask, "model_correct"] = model_correct
                updates += 1

    n_new = len(new_rows)
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    if n_new > 0 or updates > 0:
        save_pred_log(log)
    return n_new


def _grade_past_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """Fetch final scores from the MLB API for any ungraded log entries."""
    if log.empty:
        return log
    ungraded_dates = (
        log.loc[log["home_win"].isna(), "date"]
        .astype(str).str[:10].dropna().unique()
    )
    for date_str in sorted(ungraded_dates):
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&date={date_str}&hydrate=team,linescore"
            )
            data = requests.get(url, timeout=10).json()
            games = data.get("dates", [{}])[0].get("games", [])
        except Exception:
            continue
        for g in games:
            if "Final" not in g.get("status", {}).get("detailedState", ""):
                continue
            h_raw = g["teams"]["home"]["team"]["abbreviation"]
            a_raw = g["teams"]["away"]["team"]["abbreviation"]
            h = MLB_API_MAP.get(h_raw, h_raw)
            a = MLB_API_MAP.get(a_raw, a_raw)
            hs  = g["teams"]["home"].get("score")
            as_ = g["teams"]["away"].get("score")
            if hs is None or as_ is None:
                continue
            home_win = 1 if hs > as_ else 0
            mask = (
                (log["date"].astype(str).str[:10] == date_str)
                & (log["home"] == h)
                & (log["away"] == a)
                & log["home_win"].isna()
            )
            if not mask.any():
                continue
            model_home_prob = float(log.loc[mask, "model_home_prob"].iloc[0])
            log.loc[mask, "home_score"]    = hs
            log.loc[mask, "away_score"]    = as_
            log.loc[mask, "home_win"]      = home_win
            log.loc[mask, "model_correct"] = bool((model_home_prob > 0.5) == bool(home_win))
    return log


def load_ledger() -> pd.DataFrame:
    if os.path.exists(LEDGER_PATH):
        df = pd.read_csv(LEDGER_PATH, parse_dates=["date"])
        if "odds" not in df.columns:
            df["odds"] = -110
        if "notes" not in df.columns:
            df["notes"] = ""
        return df
    return pd.DataFrame(columns=LEDGER_COLS)


def save_ledger(df: pd.DataFrame):
    df.to_csv(LEDGER_PATH, index=False)


def pnl_from_odds(stake: float, odds: float, won: bool) -> float:
    """Calculate P&L from American odds."""
    if odds == 0:
        return 0.0
    if won:
        net = (odds / 100) if odds > 0 else (100 / abs(odds))
        return round(stake * net, 2)
    return round(-stake, 2)


def recalculate_ledger(df: pd.DataFrame, starting_bk: float) -> pd.DataFrame:
    """Recompute pnl and bankroll for every row based on odds and won columns."""
    df = df.copy().reset_index(drop=True)
    bk = starting_bk
    for i, row in df.iterrows():
        won = row.get("won")
        stake = float(row.get("stake", 0) or 0)
        odds  = float(row.get("odds", -110) or -110)
        if won is True or won == 1:
            p = pnl_from_odds(stake, odds, True)
        elif won is False or won == 0:
            p = pnl_from_odds(stake, odds, False)
        else:
            p = 0.0   # pending
        bk = round(bk + p, 2)
        df.at[i, "pnl"]      = p
        df.at[i, "bankroll"] = bk
    return df


def auto_log_bets(bet_rows: list, ledger: pd.DataFrame, game_date: str) -> tuple[pd.DataFrame, int]:
    """
    Add all bet_rows (from Today's Slate) to the ledger.
    Skips rows already in the ledger (same date + matchup).
    Returns (updated_ledger, n_added).
    """
    log_date = pd.Timestamp(game_date)
    log_date_str = log_date.strftime("%Y-%m-%d")

    existing_keys = set(
        zip(ledger["date"].astype(str).str[:10], ledger["matchup"])
    ) if not ledger.empty else set()

    new_rows = []
    for r in bet_rows:
        matchup = r["matchup"].split(" ")[0] + " @ " + r["matchup"].split("@ ")[-1].split(" ")[0]
        key = (log_date_str, matchup)
        if key in existing_keys:
            continue
        won = r.get("won")   # True/False/None
        odds = r.get("pick_odds", -110)
        stake = r["stake"]
        new_rows.append({
            "date":     log_date,
            "matchup":  matchup,
            "bet_side": r["pick"],
            "stake":    stake,
            "odds":     odds,
            "won":      won,
            "pnl":      0.0,
            "bankroll": 0.0,
            "notes":    "",
        })

    if not new_rows:
        return ledger, 0

    combined = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
    starting = STARTING_BANKROLL if ledger.empty else (
        float(ledger["bankroll"].iloc[0]) - float(ledger["pnl"].iloc[0])
    )
    combined = recalculate_ledger(combined, starting)
    save_ledger(combined)
    return combined, len(new_rows)


def refresh_pending(ledger: pd.DataFrame, game_date: str) -> tuple[pd.DataFrame, int]:
    """Look up scores for pending bets on game_date and fill in won/lost."""
    if ledger.empty:
        return ledger, 0

    pending_mask = ledger["won"].isna() & (ledger["date"].astype(str).str[:10] == game_date)
    if not pending_mask.any():
        return ledger, 0

    # Fetch final scores for the date
    try:
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&date={game_date}&hydrate=team,linescore")
        resp = requests.get(url, timeout=10)
        data = resp.json()
        games = data.get("dates", [{}])[0].get("games", [])
    except Exception:
        return ledger, 0

    score_map = {}
    for g in games:
        if "Final" not in g.get("status", {}).get("detailedState", ""):
            continue
        home_abbr = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
        away_abbr = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
        if not home_abbr or not away_abbr:
            continue
        h = MLB_API_MAP.get(home_abbr, home_abbr)
        a = MLB_API_MAP.get(away_abbr, away_abbr)
        hs = g.get("teams", {}).get("home", {}).get("score", 0)
        as_ = g.get("teams", {}).get("away", {}).get("score", 0)
        score_map[(a, h)] = h if hs > as_ else a

    updated = 0
    ledger = ledger.copy()
    for i, row in ledger.iterrows():
        if not pending_mask.iloc[i]:
            continue
        parts = str(row["matchup"]).split(" @ ")
        if len(parts) != 2:
            continue
        away, home = parts[0].strip(), parts[1].strip()
        winner = score_map.get((away, home))
        if winner is None:
            continue
        ledger.at[i, "won"] = (row["bet_side"].strip() == winner)
        updated += 1

    if updated:
        starting = STARTING_BANKROLL if len(ledger) == 0 else (
            float(ledger["bankroll"].iloc[0]) - float(ledger["pnl"].iloc[0])
        )
        ledger = recalculate_ledger(ledger, starting)
        save_ledger(ledger)

    return ledger, updated


def refresh_all_pending(ledger: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Refresh final scores for every date that has pending bets."""
    if ledger.empty or "won" not in ledger.columns or "date" not in ledger.columns:
        return ledger, 0

    pending_dates = (
        ledger.loc[ledger["won"].isna(), "date"]
        .astype(str)
        .str[:10]
        .dropna()
        .unique()
    )
    total_updated = 0
    for pending_date in sorted(pending_dates):
        ledger, n_updated = refresh_pending(ledger, pending_date)
        total_updated += n_updated
    return ledger, total_updated


@st.cache_data(ttl=3600)
def _update_streak_from_live(team: str, last_features_date: pd.Timestamp,
                              last_streak: float, live_logs) -> float:
    """Replay game_logs_live results on/after last_features_date to get current streak."""
    if live_logs is None or live_logs.empty:
        return last_streak
    newer = live_logs[live_logs["Date"] >= last_features_date].sort_values("Date")
    team_games = newer[(newer["home_team"] == team) | (newer["away_team"] == team)]
    streak = last_streak
    for _, row in team_games.iterrows():
        is_home = row["home_team"] == team
        won = (row["home_win"] == 1) if is_home else (row["home_win"] == 0)
        streak = (streak + 1) if won and streak > 0 else (1 if won else
                  (streak - 1) if not won and streak < 0 else -1)
    return streak


def load_team_form() -> dict[str, dict]:
    """Return latest hot/cold stats per team from features.csv."""
    features_path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(features_path):
        return {}
    df = pd.read_csv(features_path, parse_dates=["Date"])

    live_logs_path = os.path.join(DATA_DIR, "game_logs_live.csv")
    live_logs = pd.read_csv(live_logs_path, parse_dates=["Date"]) if os.path.exists(live_logs_path) else None

    form = {}
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    for team in teams:
        mask   = (df["home_team"] == team) | (df["away_team"] == team)
        subset = df[mask].sort_values("Date")
        if subset.empty:
            continue
        last   = subset.iloc[-1]
        prefix = "home" if last["home_team"] == team else "away"
        rd15   = last.get(f"{prefix}_rolling_rd", np.nan)
        rd7    = last.get(f"{prefix}_last7_rd",   np.nan)
        raw_streak = last.get(f"{prefix}_streak", 0)
        streak = _update_streak_from_live(team, pd.Timestamp(last["Date"]),
                                          float(raw_streak) if pd.notna(raw_streak) else 0.0,
                                          live_logs)
        momentum = (rd7 - rd15) if (pd.notna(rd7) and pd.notna(rd15)) else np.nan
        form[team] = {
            "rd15":     rd15,
            "rd7":      rd7,
            "streak":   int(streak),
            "momentum": momentum,
        }
    return form


def hot_cold_badge(team: str, form: dict) -> str:
    """Return 🔥 / ❄️ / empty string based on momentum."""
    if team not in form:
        return ""
    m = form[team].get("momentum", np.nan)
    if pd.isna(m):
        return ""
    if m >= 0.5:
        return "🔥"
    if m <= -0.5:
        return "❄️"
    return ""


def streak_badge(team: str, form: dict) -> str:
    if team not in form:
        return ""
    s = form[team].get("streak", 0)
    if s >= 3:
        return f"+{s}W"
    if s <= -3:
        return f"{abs(s)}L"
    return ""


@st.cache_data(ttl=3600)
def load_transactions(days_back: int = 7) -> pd.DataFrame:
    tx_path = os.path.join(DATA_DIR, "transactions_live.csv")
    if not os.path.exists(tx_path):
        return pd.DataFrame()
    df = pd.read_csv(tx_path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days_back)
    return df[df["date"] >= cutoff].reset_index(drop=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLB Win Probability",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ MLB Win Probability")

tab1, tab3, tab_acc = st.tabs(["📅 Today's Slate", "💰 Bankroll Tracker", "📊 Model Accuracy"])


# ============================================================
# TAB 1 — TODAY'S SLATE
# ============================================================
with tab1:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        slate_date = st.date_input("Date", value=date.today(), key="slate_date")
    with col_right:
        bankroll_slate = st.number_input(
            "Bankroll ($)", min_value=10.0, value=100.0, step=10.0, key="bk_slate"
        )
        kelly_frac_pct = st.slider(
            "Kelly fraction", 10, 50, 25, 5, key="kf_slate", format="%d%%"
        )
        kelly_frac_slate = kelly_frac_pct / 100
        min_conf_pct = st.slider(
            "Min confidence to show", 50, 70, 50, 1, key="mc_slate", format="%d%%"
        )
        min_conf = min_conf_pct / 100

    year = slate_date.year
    date_str = str(slate_date)

    if st.button("🔄 Load Games", key="load_slate"):
        fetch_schedule.clear()
        load_odds_for_date.clear()
        load_lineups_for_date.clear()
        get_prediction.clear()

    with st.spinner("Fetching schedule…"):
        schedule = fetch_schedule(date_str)

    # Load team form, transactions, odds, pitcher stuff, lineups, and weather once per page load
    team_form    = load_team_form()
    transactions = load_transactions(days_back=7)
    odds_df      = load_odds_for_date(date_str)
    stuff_df     = load_pitcher_stuff()
    lineups_df   = load_lineups_for_date(date_str)
    weather_df   = load_weather_data()
    has_odds     = not odds_df.empty

    if not schedule:
        st.info("No games found for this date.")
    else:
        st.markdown(f"**{len(schedule)} games on {slate_date.strftime('%A, %B %d %Y')}**")

        rows = []
        progress = st.progress(0, text="Running predictions…")
        for i, g in enumerate(schedule):
            home, away = g["home"], g["away"]
            if home not in ALL_TEAMS or away not in ALL_TEAMS:
                continue
            try:
                p_home = get_prediction(home, away, year, date_str,
                                        home_sp=g.get("home_sp", ""),
                                        away_sp=g.get("away_sp", ""))
            except Exception as _pred_err:
                st.warning(f"Prediction failed for {away}@{home}: {_pred_err}")
                continue

            p_away  = 1 - p_home
            conf    = max(p_home, p_away)

            # Get Vegas implied probs and moneylines
            pick_odds = -110
            home_ml = away_ml = -110
            vegas_p_home = vegas_p_away = np.nan
            MIN_EDGE = 0.03
            if not odds_df.empty:
                orow = odds_df[(odds_df["home_team"] == home) & (odds_df["away_team"] == away)]
                if not orow.empty:
                    r0 = orow.iloc[0]
                    home_ml = float(r0["home_ml"]) if pd.notna(r0.get("home_ml")) else -110
                    away_ml = float(r0["away_ml"]) if pd.notna(r0.get("away_ml")) else -110
                    def _imp(m): return 100/(m+100) if m>0 else abs(m)/(abs(m)+100)
                    imp_sum = _imp(home_ml) + _imp(away_ml)
                    # Only use odds for edge calc if they form a valid market (sum > 1.0)
                    if imp_sum >= 1.0:
                        t = imp_sum
                        vegas_p_home = _imp(home_ml) / t
                        vegas_p_away = _imp(away_ml) / t

            # Determine which side has edge.
            # Require the model to also agree directionally (model prob > 50%)
            # so we never bet an underdog solely because the model underestimates
            # the other side's dominance.
            edge_home = p_home - vegas_p_home if not np.isnan(vegas_p_home) else p_home - 0.5
            edge_away = p_away - vegas_p_away if not np.isnan(vegas_p_away) else p_away - 0.5

            if edge_home >= edge_away and edge_home > MIN_EDGE and p_home > 0.50:
                bet_side = home; bet_p = p_home; pick_odds = home_ml; bet_edge = edge_home
            elif edge_away > edge_home and edge_away > MIN_EDGE and p_away > 0.50:
                bet_side = away; bet_p = p_away; pick_odds = away_ml; bet_edge = edge_away
            else:
                bet_side = home if p_home > p_away else away
                bet_p = conf; pick_odds = home_ml if bet_side == home else away_ml; bet_edge = 0.0

            stake = kelly_stake(bet_p, bankroll_slate, kelly_frac_slate, ml=pick_odds) \
                    if bet_edge > MIN_EDGE else 0.0

            if pick_odds > 0:
                to_win = round(stake + stake * pick_odds / 100, 2)
            else:
                to_win = round(stake + stake * 100 / abs(pick_odds), 2) if pick_odds != 0 else 0.0

            # Hot/cold badges
            home_badge = hot_cold_badge(home, team_form) + (" " + streak_badge(home, team_form)).rstrip()
            away_badge = hot_cold_badge(away, team_form) + (" " + streak_badge(away, team_form)).rstrip()

            # Result columns (if game is final)
            final = "Final" in g["status"]
            result_str = ""
            won = pnl = home_win = None
            home_score = away_score = None
            if final and g["home_score"] is not None:
                hs, as_ = g["home_score"], g["away_score"]
                actual_winner = home if hs > as_ else away
                result_str = f"{away} {as_}–{hs} {home}"
                won = (bet_side == actual_winner)
                pnl = pnl_from_odds(stake, pick_odds, won)
                home_win   = 1 if hs > as_ else 0
                home_score = hs
                away_score = as_

            # Lineup confirmation status
            lineup_status = "⏳ TBD"
            if not lineups_df.empty:
                lrow = lineups_df[
                    (lineups_df["home_team"] == home) &
                    (lineups_df["away_team"] == away)
                ]
                if not lrow.empty:
                    h_conf = lrow.iloc[0].get("home_lineup_confirmed", False)
                    a_conf = lrow.iloc[0].get("away_lineup_confirmed", False)
                    if h_conf and a_conf:
                        lineup_status = "✅ Both out"
                    elif h_conf or a_conf:
                        lineup_status = f"½ {'Home' if h_conf else 'Away'} out"
                    else:
                        lineup_status = "⏳ TBD"

            # Vegas line
            vegas_prob = get_home_implied_prob(home, away, odds_df)
            ml_str     = get_moneyline_str(home, away, odds_df)
            if np.isnan(vegas_prob):
                model_edge = np.nan
            elif bet_side == home:
                model_edge = p_home - vegas_prob
            else:
                model_edge = p_away - (1 - vegas_prob)

            rows.append({
                "signal":     confidence_color(conf),
                "matchup":    f"{away} @ {home}",
                "home_badge": home_badge.strip(),
                "away_badge": away_badge.strip(),
                "home_sp":    g.get("home_sp", "TBD"),
                "away_sp":    g.get("away_sp", "TBD"),
                "game_time":  g.get("game_time", "TBD"),
                "live_str":   g.get("live_str",  "TBD"),
                "home%":      p_home,
                "away%":      p_away,
                "pick":       bet_side,
                "conf":       conf,
                "stake":      stake,
                "to_win":     to_win,
                "result":     result_str,
                "won":        won,
                "pnl":        pnl,
                "has_edge":   bet_edge > MIN_EDGE,
                "home":       home,
                "away":       away,
                "vegas_prob":    vegas_prob,
                "ml_str":        ml_str,
                "model_edge":    model_edge,
                "lineup_status": lineup_status,
                "pick_odds":     pick_odds,
                "home_score":    home_score,
                "away_score":    away_score,
                "home_win":      home_win,
            })
            progress.progress((i + 1) / len(schedule), text=f"Predicting {away} @ {home}…")

        progress.empty()

        # Persist all predictions (and scores for finished games) silently
        log_or_update_predictions(rows, date_str)

        show_rows = [r for r in rows if r["conf"] >= min_conf]
        show_rows.sort(key=lambda r: -r["conf"])

        # Summary metrics
        bets = [r for r in show_rows if r["has_edge"] and r["stake"] > 0]
        total_stake = sum(r["stake"] for r in bets)
        finished = [r for r in bets if r["pnl"] is not None]
        total_pnl = sum(r["pnl"] for r in finished)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Games", len(schedule))
        m2.metric("With edge", len(bets))
        m3.metric("Total at risk", f"${total_stake:.2f}")
        if finished:
            record = f"{sum(1 for r in finished if r['won'])}-{sum(1 for r in finished if not r['won'])}"
            m4.metric("P&L (final games)", f"${total_pnl:+.2f}", delta=record)

        # ── Main predictions table ────────────────────────────────
        display = []
        has_finals = any("Final" in g.get("status", "") for g in schedule)
        for r in show_rows:
            away_str = f"{r['away']} {r['away_badge']}".strip()
            home_str = f"{r['home']} {r['home_badge']}".strip()
            row = {
                "":          r["signal"],
                "Live":      r.get("live_str", r["game_time"]),
                "Matchup":   f"{away_str} @ {home_str}",
                "Pitchers":  f"{r['away_sp']} vs {r['home_sp']}",
                "Lineups":   r["lineup_status"],
                "Home%":     f"{r['home%']:.1%}",
                "Away%":     f"{r['away%']:.1%}",
                "Pick":      r["pick"],
                "Conf":      f"{r['conf']:.1%}",
                "Stake":     f"${r['stake']:.2f}" if r["has_edge"] else "—",
                "Payout":    f"${r['to_win']:.2f}" if r["has_edge"] else "—",
            }
            if has_odds and not np.isnan(r.get("vegas_prob", np.nan)):
                row["Vegas"] = r["ml_str"]
                vegas_pick_prob = r["vegas_prob"] if r["pick"] == r["home"] else (1 - r["vegas_prob"])
                row["Vegas Pick%"] = f"{vegas_pick_prob:.1%}"
                edge = r["model_edge"]
                row["Edge"] = f"{edge:+.1%}" if not np.isnan(edge) else "—"
            if has_finals:
                row["Result"] = r["result"] if r["result"] else "Pending"
                row["P&L"]    = f"${r['pnl']:+.2f}" if r["pnl"] is not None else "—"
            display.append(row)

        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

        if bets:
            odds_note = "  |  Edge = model% − Vegas%" if has_odds else "  |  Set ODDS_API_KEY for Vegas lines"
            st.caption(
                f"🟢 ≥65% confidence  🟡 58–65%  ⚪ 53–58%  |  "
                f"🔥 heating up  ❄️ cooling down  |  "
                f"Break-even at -110: {BREAKEVEN:.1%}  |  Kelly fraction: {kelly_frac_slate:.0%}"
                + odds_note
            )

        # ── Auto-log today's bets ─────────────────────────────────
        if bets:
            if st.button("📥 Log Today's Bets to Tracker", key="auto_log"):
                current_ledger = load_ledger()
                updated_ledger, n = auto_log_bets(bets, current_ledger, date_str)
                if n > 0:
                    st.success(f"Added {n} bet(s) to the Bankroll Tracker. Switch to the 💰 tab to view.")
                else:
                    st.info("All of today's bets are already in the tracker.")

        # ── Game summaries ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Game Summaries")
        for r in show_rows:
            edge_flag = "🎯 " if r["has_edge"] else ""
            pick_pct  = f"{r['conf']:.1%}"
            label     = f"{edge_flag}{r['away']} @ {r['home']}  —  {r['live_str']}  |  Pick: **{r['pick']}** {pick_pct}"
            with st.expander(label, expanded=False):
                home_stuff = get_sp_stats(r["home_sp"], r["home"], year, stuff_df)
                away_stuff = get_sp_stats(r["away_sp"], r["away"], year, stuff_df)
                weather    = get_weather_for_game(r["home"], date_str, weather_df)

                # ── Key factors ──────────────────────────────────
                factors = generate_key_factors(r, home_stuff, away_stuff, team_form, weather, year)
                st.markdown("**Why this pick:**")
                for f in factors:
                    st.markdown(f"&nbsp;&nbsp;• {f}")

                st.markdown("")
                c1, c2, c3 = st.columns(3)

                # ── Pitching matchup ─────────────────────────────
                with c1:
                    st.markdown("**Pitching Matchup**")
                    for side, sp_name, stuff in [
                        (r["away"], r["away_sp"], away_stuff),
                        (r["home"], r["home_sp"], home_stuff),
                    ]:
                        hand   = stuff.get("hand", "?")
                        xfip   = f"{stuff['xFIP']:.2f}" if "xFIP" in stuff else "—"
                        kpct   = f"{stuff['K_pct']*100:.1f}%" if "K_pct" in stuff else "—"
                        swstr  = f"{stuff['SwStr_pct']*100:.1f}%" if "SwStr_pct" in stuff else "—"
                        fbv    = f"{stuff['FBv']:.1f}" if "FBv" in stuff else "—"
                        gs     = f"{int(stuff['GS'])} GS" if "GS" in stuff else ""
                        st.markdown(
                            f"**{side}** — {sp_name} ({hand})  \n"
                            f"xFIP: `{xfip}` &nbsp; K%: `{kpct}` &nbsp; SwStr%: `{swstr}`  \n"
                            f"FB: `{fbv} mph` &nbsp; {gs}"
                        )

                # ── Team form ────────────────────────────────────
                with c2:
                    st.markdown("**Team Form**")
                    for team in [r["away"], r["home"]]:
                        f = team_form.get(team, {})
                        streak  = f.get("streak", 0)
                        rd15    = f.get("rd15", np.nan)
                        rd7     = f.get("rd7",  np.nan)
                        mom     = f.get("momentum", np.nan)
                        badge   = "🔥" if (not pd.isna(mom) and mom >= 0.5) else ("❄️" if (not pd.isna(mom) and mom <= -0.5) else "")
                        streak_str = (f"+{streak}W" if streak >= 1 else f"{abs(streak)}L") if streak != 0 else "even"
                        rd15_s  = f"{rd15:+.2f}" if pd.notna(rd15) else "—"
                        rd7_s   = f"{rd7:+.2f}"  if pd.notna(rd7)  else "—"
                        st.markdown(
                            f"**{team}** {badge}  \n"
                            f"Streak: `{streak_str}` &nbsp; RunDiff 15g: `{rd15_s}` &nbsp; 7g: `{rd7_s}`"
                        )

                # ── Conditions & edge ────────────────────────────
                with c3:
                    st.markdown("**Conditions & Edge**")
                    # Weather
                    if weather and r["home"] not in FULL_DOME:
                        temp = weather.get("temp_f")
                        spd  = weather.get("wind_speed_mph")
                        deg  = weather.get("wind_dir_deg")
                        hum  = weather.get("humidity_pct")
                        wtcf = wind_to_cf(spd or 0, deg or 0, r["home"]) if spd and deg else 0
                        wcf_str = f"↑ out +{wtcf:.1f}" if wtcf >= 2 else (f"↓ in {wtcf:.1f}" if wtcf <= -2 else "neutral")
                        st.markdown(
                            f"🌡️ `{temp:.0f}°F`" + (f"  💧 `{hum:.0f}%`" if hum else "") +
                            (f"  🌬️ `{spd:.0f} mph` ({wcf_str})" if spd else "")
                        )
                    elif r["home"] in FULL_DOME:
                        st.markdown("🏟️ Dome — weather neutral")
                    else:
                        st.markdown("🌤️ Weather not available")

                    st.markdown("")
                    # Model vs Vegas
                    hp  = r["home%"]
                    ap  = r["away%"]
                    vp  = r.get("vegas_prob", np.nan)
                    st.markdown(
                        f"Model: `{r['away']} {ap:.1%}` vs `{r['home']} {hp:.1%}`  \n"
                        + (f"Vegas:  `{r['away']} {1-vp:.1%}` vs `{r['home']} {vp:.1%}`  \n" if not np.isnan(vp) else "")
                        + (f"Edge on **{r['pick']}**: `{r['model_edge']:+.1%}`" if not np.isnan(r.get("model_edge") or np.nan) else "")
                    )

        # ── Roster moves (last 7 days) for today's teams ──────────
        if not transactions.empty:
            today_teams = {r["home"] for r in rows} | {r["away"] for r in rows}
            relevant_tx = transactions[transactions["team"].isin(today_teams)]
            notable_codes = {"IL", "SFA", "TR", "RM", "DES"}
            relevant_tx = relevant_tx[relevant_tx["type_code"].isin(notable_codes)]
            if not relevant_tx.empty:
                with st.expander(f"🗒️ Recent roster moves — today's teams ({len(relevant_tx)} moves)", expanded=False):
                    tx_display = relevant_tx[["date", "team", "type_label", "player", "description"]].copy()
                    tx_display["date"] = tx_display["date"].dt.strftime("%b %d")
                    tx_display.columns = ["Date", "Team", "Move", "Player", "Description"]
                    st.dataframe(tx_display, use_container_width=True, hide_index=True)




# ============================================================
# TAB 3 — BANKROLL TRACKER
# ============================================================
with tab3:
    st.subheader("💰 Bankroll Tracker")

    ledger = load_ledger()

    # Derive starting bankroll from first row or default
    if ledger.empty:
        first_bk = STARTING_BANKROLL
    else:
        first_bk = float(ledger["bankroll"].iloc[0]) - float(ledger["pnl"].iloc[0])

    # ── Action buttons row ─────────────────────────────────────
    ac1, ac2, ac3 = st.columns([2, 2, 1])
    with ac1:
        if st.button("🔄 Refresh Results (update pending bets)", key="refresh_pending"):
            ledger, n_updated = refresh_all_pending(ledger)
            if n_updated:
                st.success(f"Updated {n_updated} bet(s) with final scores.")
            else:
                st.info("No pending bets with final scores to update.")
            st.rerun()
    with ac2:
        with st.expander("➕ Manually log a bet"):
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                log_date    = st.date_input("Date", value=date.today(), key="log_date")
                log_matchup = st.text_input("Matchup", placeholder="BOS @ NYY")
            with lc2:
                log_side  = st.text_input("Team you bet", placeholder="NYY")
                log_stake = st.number_input("Stake ($)", min_value=0.01, value=5.00, step=0.50)
                log_odds  = st.number_input("Odds", value=-110, step=5)
            with lc3:
                log_won   = st.radio("Result", ["Win", "Loss", "Pending"], horizontal=True)
                log_notes = st.text_input("Notes (optional)")
            if st.button("Save bet", key="save_bet"):
                won_val = True if log_won == "Win" else (False if log_won == "Loss" else None)
                new_row = pd.DataFrame([{
                    "date": pd.Timestamp(log_date), "matchup": log_matchup,
                    "bet_side": log_side, "stake": log_stake, "odds": log_odds,
                    "won": won_val, "pnl": 0.0, "bankroll": 0.0, "notes": log_notes,
                }])
                ledger = pd.concat([ledger, new_row], ignore_index=True)
                ledger = recalculate_ledger(ledger, first_bk)
                save_ledger(ledger)
                st.success("Bet logged!")
                st.rerun()

    if not ledger.empty:
        completed = ledger[ledger["won"].notna()]
        pending   = ledger[ledger["won"].isna()]
        wins      = completed["won"].sum()
        losses    = len(completed) - wins
        net_pnl   = ledger["pnl"].sum()
        final_bk  = float(ledger["bankroll"].iloc[-1])
        roi       = net_pnl / ledger["stake"].sum() * 100 if ledger["stake"].sum() > 0 else 0

        # ── Summary metrics ───────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Starting", f"${first_bk:.2f}")
        m2.metric("Bankroll", f"${final_bk:.2f}", delta=f"${net_pnl:+.2f}")
        m3.metric("Record", f"{int(wins)}-{int(losses)}")
        m4.metric("Win rate", f"{wins/len(completed):.1%}" if len(completed) else "—")
        m5.metric("ROI", f"{roi:+.1f}%")
        m6.metric("Pending", f"{len(pending)} bets")

        # ── Editable bet history ───────────────────────────────
        st.markdown("#### Bet History")
        st.caption("Odds column uses American format (e.g. -110, +130). P&L updates when you click Save Changes.")

        edit_df = ledger.copy()
        edit_df["date"] = edit_df["date"].dt.strftime("%Y-%m-%d")
        edit_df["Result"] = edit_df["won"].map(
            {True: "✅ Win", False: "❌ Loss", None: "⏳ Pending"}
        ).fillna("⏳ Pending")
        edit_df["P&L"] = edit_df["pnl"].map(lambda x: f"${x:+.2f}" if x != 0 else "—")

        display_cols = ["date", "matchup", "bet_side", "stake", "odds", "Result", "P&L", "notes"]
        rename_map   = {"date": "Date", "matchup": "Matchup", "bet_side": "Bet",
                        "stake": "Stake", "odds": "Odds", "notes": "Notes"}
        display_edit = edit_df[display_cols].rename(columns=rename_map)
        display_edit["Notes"] = display_edit["Notes"].fillna("").astype(str)

        edited = st.data_editor(
            display_edit[::-1].reset_index(drop=True),
            num_rows="dynamic",
            column_config={
                "Date":    st.column_config.TextColumn("Date", help="YYYY-MM-DD"),
                "Matchup": st.column_config.TextColumn("Matchup", help="e.g. BOS @ NYY"),
                "Bet":     st.column_config.TextColumn("Bet", help="Team abbreviation you bet"),
                "Odds":    st.column_config.NumberColumn("Odds", help="American odds (-110, +130)", step=5),
                "Stake":   st.column_config.NumberColumn("Stake ($)", format="%.2f"),
                "Result":  st.column_config.SelectboxColumn("Result",
                               options=["✅ Win", "❌ Loss", "⏳ Pending"]),
                "P&L":     st.column_config.TextColumn("P&L", disabled=True),
                "Notes":   st.column_config.TextColumn("Notes"),
            },
            use_container_width=True,
            hide_index=True,
            key="ledger_editor",
        )

        # Live P&L preview from current editor state (updates as you edit)
        def _live_pnl(row) -> float:
            stake  = float(row.get("Stake") or 0)
            odds   = float(row.get("Odds") or -110)
            result = row.get("Result", "⏳ Pending")
            if result == "✅ Win":
                net = odds / 100 if odds > 0 else 100 / abs(odds)
                return round(stake * net, 2)
            if result == "❌ Loss":
                return round(-stake, 2)
            return 0.0

        live_total  = sum(_live_pnl(row) for _, row in edited.iterrows())
        live_staked = sum(float(r.get("Stake") or 0) for _, r in edited.iterrows())
        live_wins   = sum(1 for _, r in edited.iterrows() if r.get("Result") == "✅ Win")
        live_losses = sum(1 for _, r in edited.iterrows() if r.get("Result") == "❌ Loss")
        lp1, lp2, lp3, lp4 = st.columns(4)
        lp1.metric("Total Staked", f"${live_staked:.2f}")
        lp2.metric("Live P&L", f"${live_total:+.2f}")
        lp3.metric("Wins", live_wins)
        lp4.metric("Losses", live_losses)

        if st.button("💾 Save Changes", key="save_edits"):
            # Reverse back to original order
            edited = edited[::-1].reset_index(drop=True)
            edited = edited.dropna(how="all").reset_index(drop=True)
            required_cols = ["Date", "Matchup", "Bet"]
            missing_required = edited[required_cols].isna().any(axis=1)
            for col in required_cols:
                missing_required |= edited[col].astype(str).str.strip().eq("")
            if missing_required.any():
                st.error("Remove blank added rows before saving changes.")
                st.stop()
            parsed_dates = pd.to_datetime(edited["Date"], errors="coerce")
            if parsed_dates.isna().any():
                st.error("One or more ledger rows has an invalid date.")
                st.stop()
            # Map display columns back to internal schema
            new_ledger = pd.DataFrame({
                "date":     parsed_dates,
                "matchup":  edited["Matchup"],
                "bet_side": edited["Bet"],
                "stake":    pd.to_numeric(edited["Stake"], errors="coerce"),
                "odds":     pd.to_numeric(edited["Odds"], errors="coerce").fillna(-110),
                "won":      edited["Result"].map(
                                {"✅ Win": True, "❌ Loss": False, "⏳ Pending": None}),
                "pnl":      0.0,
                "bankroll": 0.0,
                "notes":    edited.get("Notes", ""),
            })
            new_ledger = recalculate_ledger(new_ledger, first_bk)
            save_ledger(new_ledger)
            st.success("Changes saved and bankroll recalculated!")
            st.rerun()

        if st.button("🗑️ Clear ALL data", key="clear_ledger"):
            os.remove(LEDGER_PATH)
            st.rerun()
    else:
        st.info("No bets yet. Use **📥 Log Today's Bets** on the Today's Slate tab, or add one manually above.")


# ============================================================
# TAB — MODEL ACCURACY
# ============================================================
with tab_acc:
    st.subheader("📊 Model Accuracy & Calibration")
    st.caption(
        "Logs every game the model evaluates (not just bets). "
        "Auto-populated when you load Today's Slate."
    )

    pred_log = load_pred_log()

    if st.button("🔄 Grade Ungraded Past Games", key="grade_past"):
        with st.spinner("Fetching final scores…"):
            pred_log = _grade_past_predictions(pred_log)
            save_pred_log(pred_log)
        st.success("Done.")
        st.rerun()

    if pred_log.empty:
        st.info("No predictions logged yet. Load Today's Slate to start tracking.")
        st.stop()

    graded = pred_log[pred_log["model_correct"].notna()].copy()

    if graded.empty:
        st.info("No completed games graded yet. Try **Grade Ungraded Past Games**.")
        st.stop()

    graded["home_win"]      = graded["home_win"].astype(float)
    graded["model_correct"] = graded["model_correct"].astype(float)

    acc = graded["model_correct"].mean()
    n   = len(graded)
    correct = int(graded["model_correct"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{acc:.1%}")
    m2.metric("Games Graded", n)
    m3.metric("Correct", correct)
    m4.metric("Incorrect", n - correct)

    st.markdown("---")

    # ── Daily breakdown ───────────────────────────────────────
    st.markdown("#### Accuracy by Date")
    daily = (
        graded.groupby(graded["date"].dt.strftime("%Y-%m-%d"))
        .agg(Games=("model_correct", "count"), Correct=("model_correct", "sum"))
        .assign(Accuracy=lambda d: d["Correct"] / d["Games"])
        .reset_index()
        .rename(columns={"date": "Date"})
        .sort_values("Date", ascending=False)
    )
    daily["Accuracy"] = daily["Accuracy"].map(lambda x: f"{x:.1%}")
    daily["Correct"]  = daily["Correct"].astype(int)
    st.dataframe(daily, use_container_width=True, hide_index=True)

    # ── Calibration ───────────────────────────────────────────
    st.markdown("#### Calibration")
    st.caption(
        "When the model gives the home team X% chance, how often do they actually win? "
        "A well-calibrated model should track the diagonal."
    )
    bins   = [0, .45, .50, .55, .60, .65, .70, 1.0]
    labels = ["<45%", "45–50%", "50–55%", "55–60%", "60–65%", "65–70%", ">70%"]
    graded["prob_bin"] = pd.cut(
        graded["model_home_prob"], bins=bins, labels=labels, right=True
    )
    cal = (
        graded.groupby("prob_bin", observed=True)
        .agg(Games=("home_win", "count"), Actual_Win_Rate=("home_win", "mean"))
        .reset_index()
        .rename(columns={"prob_bin": "Model Prob Bin", "Actual_Win_Rate": "Actual Win Rate"})
    )
    cal["Actual Win Rate"] = cal["Actual Win Rate"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    st.dataframe(cal, use_container_width=True, hide_index=True)

    # ── Recent predictions ────────────────────────────────────
    st.markdown("#### Recent Predictions")
    recent = (
        pred_log.sort_values("date", ascending=False)
        .head(50)
        .copy()
    )
    recent["Date"]       = recent["date"].dt.strftime("%Y-%m-%d")
    recent["Matchup"]    = recent["away"] + " @ " + recent["home"]
    recent["Pitchers"]   = recent["away_sp"] + " vs " + recent["home_sp"]
    recent["Model Home"] = recent["model_home_prob"].map(lambda x: f"{x:.1%}")
    recent["Vegas Home"] = recent["vegas_home_prob"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    recent["Pick"]       = recent["pick"]
    recent["Score"]      = recent.apply(
        lambda r: (f"{r['away']} {int(r['away_score'])}–{int(r['home_score'])} {r['home']}"
                   if pd.notna(r.get("home_score")) else "—"),
        axis=1,
    )
    recent["Correct"] = recent["model_correct"].map(
        {1.0: "✅", 0.0: "❌", None: "⏳"}
    ).fillna("⏳")

    st.dataframe(
        recent[["Date","Matchup","Pitchers","Model Home","Vegas Home","Pick","Score","Correct"]],
        use_container_width=True,
        hide_index=True,
    )
