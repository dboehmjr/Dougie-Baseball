"""
Daily data refresh — run once per day (ideally ~6 AM before the day's games).

What it does:
  1. Fetches completed game results from the MLB Stats API
  2. Fetches the last 7 days of roster transactions
  3. Refreshes FanGraphs SP pitch stuff (incremental, current year only)
  4. Refreshes team batting splits vs LHP/RHP (current year)
  5. Re-runs feature engineering so rolling windows (incl. hot/cold) are current
  6. Prints a concise summary of what changed

Usage:
    python daily_update.py               # updates through yesterday
    python daily_update.py --full        # re-fetches entire 2026 season
    python daily_update.py --date 2026-04-20   # update through a specific date
    python daily_update.py --skip-features     # skip feature engineering rebuild
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_RESULTS_PATH  = os.path.join(DATA_DIR, "game_logs_live.csv")
TRANSACTIONS_PATH  = os.path.join(DATA_DIR, "transactions_live.csv")
SEASON_START       = date(2026, 3, 18)

# MLB Stats API team abbreviation → model abbreviation
MLB_ABBREV_MAP = {
    "WSH": "WSN", "SD": "SDP", "TB": "TBR",
    "KC": "KCR", "AZ": "ARI", "SF": "SFG",
    "ATH": "OAK", "CWS": "CHW",
}

# Transaction type codes we care about
TRANSACTION_TYPES = {
    "IL":  "🏥 IL",
    "RM":  "✅ Activated",
    "TR":  "🔄 Trade",
    "DES": "📋 DFA",
    "SFA": "⬆️ Called Up",
    "SCR": "⬇️ Optioned",
    "CLW": "📦 Claimed",
    "REL": "🚪 Released",
}


def _abbrev(raw: str) -> str:
    return MLB_ABBREV_MAP.get(raw.upper(), raw.upper())


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# 1. Game results
# ---------------------------------------------------------------------------

def fetch_results_for_date(game_date: date) -> list[dict]:
    """Return completed game rows for a single date."""
    url    = "https://statsapi.mlb.com/api/v1/schedule"
    data   = _get(url, {"sportId": 1, "date": str(game_date),
                         "hydrate": "team,linescore"})
    games  = data.get("dates", [{}])[0].get("games", [])
    rows   = []
    for g in games:
        if "Final" not in g.get("status", {}).get("detailedState", ""):
            continue
        home_raw  = g["teams"]["home"]["team"]["abbreviation"]
        away_raw  = g["teams"]["away"]["team"]["abbreviation"]
        home      = _abbrev(home_raw)
        away      = _abbrev(away_raw)
        h_score   = g["teams"]["home"].get("score")
        a_score   = g["teams"]["away"].get("score")
        if h_score is None or a_score is None:
            continue
        rows.append({
            "Date":       str(game_date),
            "year":       game_date.year,
            "home_team":  home,
            "away_team":  away,
            "home_win":   int(h_score > a_score),
            "home_runs":  int(h_score),
            "away_runs":  int(a_score),
        })
    return rows


def update_game_logs(through: date, full_season: bool = False) -> int:
    """
    Fetch any missing game results from the MLB Stats API and append to
    game_logs_live.csv.  Returns the number of new rows added.
    """
    existing = pd.DataFrame()
    if os.path.exists(LIVE_RESULTS_PATH) and not full_season:
        existing = pd.read_csv(LIVE_RESULTS_PATH, parse_dates=["Date"])

    already = set(
        zip(existing["Date"].astype(str), existing["home_team"])
    ) if not existing.empty else set()

    fetch_from = SEASON_START if full_season else (
        (pd.to_datetime(existing["Date"].max()).date() + timedelta(days=1))
        if not existing.empty else SEASON_START
    )

    all_rows = []
    d = fetch_from
    while d <= through:
        rows = fetch_results_for_date(d)
        new  = [r for r in rows if (str(r["Date"]), r["home_team"]) not in already]
        all_rows.extend(new)
        d += timedelta(days=1)
        time.sleep(0.15)

    if not all_rows:
        return 0

    new_df   = pd.DataFrame(all_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["Date", "home_team"]).drop_duplicates(
        subset=["Date", "home_team", "away_team"]
    )
    combined.to_csv(LIVE_RESULTS_PATH, index=False)
    return len(all_rows)


# ---------------------------------------------------------------------------
# 2. Roster transactions
# ---------------------------------------------------------------------------

def fetch_transactions(days_back: int = 7) -> pd.DataFrame:
    """
    Fetch roster transactions from the MLB Stats API for the last `days_back`
    days.  Returns a cleaned DataFrame.
    """
    end_date   = date.today()
    start_date = end_date - timedelta(days=days_back)
    url        = "https://statsapi.mlb.com/api/v1/transactions"
    data       = _get(url, {
        "sportId":   1,
        "startDate": str(start_date),
        "endDate":   str(end_date),
    })

    rows = []
    for t in data.get("transactions", []):
        type_code = t.get("typeCode", "")
        type_label = TRANSACTION_TYPES.get(type_code, type_code)
        if not type_label:
            continue

        from_team_raw = (t.get("fromTeam") or {}).get("abbreviation", "")
        to_team_raw   = (t.get("toTeam")   or {}).get("abbreviation", "")
        from_team = _abbrev(from_team_raw) if from_team_raw else ""
        to_team   = _abbrev(to_team_raw)   if to_team_raw   else ""
        team      = from_team or to_team

        player    = (t.get("person") or {}).get("fullName", "Unknown")
        desc      = t.get("typeDesc") or type_label
        tx_date   = t.get("effectiveDate") or t.get("date", "")

        rows.append({
            "date":       tx_date[:10] if tx_date else "",
            "team":       team,
            "player":     player,
            "type_code":  type_code,
            "type_label": type_label,
            "description": desc,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "team", "player", "type_code", "type_label", "description"]
    )

    if not df.empty:
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
        df.to_csv(TRANSACTIONS_PATH, index=False)

    return df


# ---------------------------------------------------------------------------
# 3. SP pitch stuff (FanGraphs via pybaseball) — incremental, current year
# ---------------------------------------------------------------------------

def run_pitcher_stuff_fetch() -> bool:
    """Fetch/refresh FanGraphs pitcher stuff for the current year."""
    try:
        from fetch_pitcher_stuff import fetch_pitcher_stuff
        from datetime import date as _date
        year = _date.today().year
        df = fetch_pitcher_stuff(start_year=year, end_year=year)
        print(f"  SP stuff refreshed: {len(df):,} total pitcher-seasons")
        return True
    except Exception as exc:
        print(f"  SP stuff fetch FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 3b. Statcast batting metrics (Baseball Savant via pybaseball)
# ---------------------------------------------------------------------------

def run_statcast_batting_fetch() -> bool:
    """Fetch/refresh Statcast team batting metrics for the current year."""
    try:
        from fetch_statcast_batting import fetch_statcast_batting
        from datetime import date as _date
        year = _date.today().year
        df = fetch_statcast_batting(start_year=year, end_year=year)
        print(f"  Statcast batting refreshed: {len(df):,} total team-seasons")
        return True
    except Exception as exc:
        print(f"  Statcast batting fetch FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 4. Team batting splits (MLB Stats API) — current year only
# ---------------------------------------------------------------------------

def run_splits_fetch() -> bool:
    """Fetch/refresh L/R batting splits for the current year."""
    try:
        from fetch_splits import fetch_all_splits
        from datetime import date as _date
        year = _date.today().year
        df = fetch_all_splits(start_year=year, end_year=year)
        print(f"  Team splits refreshed: {len(df):,} total team-seasons")
        return True
    except Exception as exc:
        print(f"  Team splits fetch FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 5. Weather refresh
# ---------------------------------------------------------------------------

def run_weather_refresh() -> bool:
    try:
        from fetch_weather import refresh_recent_weather
        refresh_recent_weather()
        return True
    except Exception as exc:
        print(f"  Weather refresh FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 6. IL count refresh
# ---------------------------------------------------------------------------

def run_il_refresh() -> bool:
    try:
        from fetch_il_data import fetch_all_il_counts, get_il_count, STATSAPI_TO_ABBREV
        from datetime import date as _date
        import time as _time

        today = _date.today()
        df = fetch_all_il_counts(start_year=2015, end_year=today.year)

        # Also write today's live counts into il_counts.csv so feature
        # engineering always has a fresh snapshot for the current date.
        il_path = os.path.join(DATA_DIR, "il_counts.csv")
        existing = pd.read_csv(il_path, parse_dates=["date"]) if os.path.exists(il_path) else pd.DataFrame()
        today_str = str(today)
        already_today = (
            not existing.empty
            and (existing["date"].dt.date == today).any()
        )
        if not already_today:
            print(f"  Writing today's live IL snapshot ({today_str})…")
            rows = []
            for team_id, abbrev in STATSAPI_TO_ABBREV.items():
                count = get_il_count(team_id, today_str)
                rows.append({"date": today_str, "team": abbrev, "il_count": count})
                _time.sleep(0.1)
            today_df = pd.DataFrame(rows)
            today_df["date"] = pd.to_datetime(today_df["date"])
            combined = pd.concat([existing, today_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "team"]).sort_values(["team", "date"])
            combined.to_csv(il_path, index=False)
            print(f"  Today's IL snapshot added ({len(rows)} teams)")

        print(f"  IL counts refreshed: {len(df):,} rows")
        return True
    except Exception as exc:
        print(f"  IL refresh FAILED: {exc}")
        return False


def run_il_quality_refresh() -> bool:
    """Refresh WAR-weighted IL quality scores."""
    try:
        from fetch_il_data import fetch_all_il_quality
        from datetime import date as _date
        war_path = os.path.join(DATA_DIR, "player_war.csv")
        if not os.path.exists(war_path):
            print("  player_war.csv not found — run fetch_player_war.py first")
            return False
        war_df = pd.read_csv(war_path)
        df = fetch_all_il_quality(war_df, start_year=2015, end_year=_date.today().year)
        print(f"  IL quality refreshed: {len(df):,} rows")
        return True
    except Exception as exc:
        print(f"  IL quality refresh FAILED: {exc}")
        return False


def run_player_war_refresh() -> bool:
    """Refresh individual player WAR from Baseball Reference."""
    try:
        from fetch_player_war import fetch_player_war
        df = fetch_player_war(start_year=2014)  # 2014 so 2015 games have prior-year data
        print(f"  Player WAR refreshed: {len(df):,} player-seasons")
        return True
    except Exception as exc:
        print(f"  Player WAR refresh FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 7. Historical odds refresh
# ---------------------------------------------------------------------------

def run_odds_refresh() -> bool:
    try:
        from fetch_historical_odds import fetch_historical_odds
        df = fetch_historical_odds()
        print(f"  Historical odds refreshed: {len(df):,} rows")
        return True
    except Exception as exc:
        print(f"  Odds refresh FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 8. Pitcher handedness backfill
# ---------------------------------------------------------------------------

def run_handedness_backfill() -> bool:
    try:
        from fetch_pitcher_stuff import backfill_handedness
        backfill_handedness()
        return True
    except Exception as exc:
        print(f"  Handedness backfill FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# 9. Re-run feature engineering
# ---------------------------------------------------------------------------

def run_feature_engineering() -> bool:
    """Re-run feature_engineering.py as a subprocess. Returns True on success."""
    script = os.path.join(os.path.dirname(__file__), "feature_engineering.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True,
        cwd=os.path.dirname(__file__)
    )
    if result.returncode != 0:
        # Write full stderr to a log file so it isn't truncated in the terminal
        log_path = os.path.join(os.path.dirname(__file__), "logs", "feat_eng_error.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(result.stderr)
        print(f"  Feature engineering FAILED (full error → {log_path}):\n"
              f"  {result.stderr[-500:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# 4. Summary helpers
# ---------------------------------------------------------------------------

def print_hot_cold_summary(n: int = 5) -> None:
    """Print the hottest and coldest teams based on momentum (last7 vs last15 run diff)."""
    features_path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(features_path):
        return

    df    = pd.read_csv(features_path, parse_dates=["Date"])
    today = pd.Timestamp.today().normalize()

    # Get latest snapshot per team
    records = []
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    for team in teams:
        mask   = ((df["home_team"] == team) | (df["away_team"] == team))
        subset = df[mask].sort_values("Date")
        if subset.empty:
            continue
        last   = subset.iloc[-1]
        prefix = "home" if last["home_team"] == team else "away"
        rd15   = last.get(f"{prefix}_rolling_rd", float("nan"))
        rd7    = last.get(f"{prefix}_last7_rd",   float("nan"))
        streak = last.get(f"{prefix}_streak",     0)
        if pd.isna(rd15) or pd.isna(rd7):
            continue
        momentum = rd7 - rd15
        records.append({
            "team": team, "rd15": rd15, "rd7": rd7,
            "momentum": momentum, "streak": int(streak),
        })

    if not records:
        return

    rec_df = pd.DataFrame(records).sort_values("momentum", ascending=False)

    def streak_str(s):
        return f"+{s}W" if s > 0 else f"{abs(s)}L" if s < 0 else "—"

    print(f"\n{'─'*45}")
    print(f"  🔥 Hottest teams (gaining momentum):")
    for _, r in rec_df.head(n).iterrows():
        print(f"    {r['team']:<4}  7g RD: {r['rd7']:+.2f}  15g RD: {r['rd15']:+.2f}  "
              f"Δ: {r['momentum']:+.2f}  Streak: {streak_str(r['streak'])}")
    print(f"\n  ❄️  Coldest teams (losing momentum):")
    for _, r in rec_df.tail(n).iloc[::-1].iterrows():
        print(f"    {r['team']:<4}  7g RD: {r['rd7']:+.2f}  15g RD: {r['rd15']:+.2f}  "
              f"Δ: {r['momentum']:+.2f}  Streak: {streak_str(r['streak'])}")


def print_notable_transactions(df: pd.DataFrame) -> None:
    """Print IL placements, call-ups, and trades from the last 7 days."""
    if df.empty:
        return
    notable_codes = {"IL", "SFA", "TR", "RM", "DES"}
    notable = df[df["type_code"].isin(notable_codes)].head(20)
    if notable.empty:
        return
    print(f"\n{'─'*45}")
    print(f"  Recent roster moves (last 7 days):")
    for _, r in notable.iterrows():
        print(f"    {r['date']}  {r['team']:<4}  {r['type_label']}  {r['player']}  — {r['description']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily MLB data refresh")
    parser.add_argument("--full",  action="store_true",
                        help="Re-fetch entire current season (slow)")
    parser.add_argument("--date",  default=None,
                        help="Update through this date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering rebuild")
    args = parser.parse_args()

    if args.date:
        try:
            through = date.fromisoformat(args.date)
        except ValueError:
            print(f"ERROR: --date must be YYYY-MM-DD, got '{args.date}'")
            sys.exit(1)
    else:
        through = date.today() - timedelta(days=1)

    print(f"\n{'='*45}")
    print(f"  MLB DAILY UPDATE — through {through}")
    print(f"{'='*45}")

    # Step 1: game results
    print("\n[1/5] Fetching game results…")
    try:
        n_new = update_game_logs(through, full_season=args.full)
        live  = pd.read_csv(LIVE_RESULTS_PATH)
        print(f"  New games added : {n_new}")
        print(f"  Total live games: {len(live):,}")
    except Exception as e:
        print(f"  ERROR fetching results: {e}")

    # Step 2: transactions
    print("\n[2/5] Fetching roster transactions…")
    try:
        tx_df = fetch_transactions(days_back=7)
        print(f"  Transactions fetched: {len(tx_df)}")
        print_notable_transactions(tx_df)
    except Exception as e:
        print(f"  ERROR fetching transactions: {e}")
        tx_df = pd.DataFrame()

    # Step 3: SP pitch stuff (FanGraphs)
    print("\n[3/9] Refreshing SP pitch stuff (FanGraphs)…")
    run_pitcher_stuff_fetch()

    # Step 3b: Statcast batting (barrel rate, hard hit%)
    print("\n[3b/9] Refreshing Statcast batting metrics (Baseball Savant)…")
    run_statcast_batting_fetch()

    # Step 4: batting splits
    print("\n[4/9] Refreshing team batting splits (L/R)…")
    run_splits_fetch()

    # Step 5: weather refresh
    print("\n[5/9] Refreshing weather data…")
    run_weather_refresh()

    # Step 6: IL counts + quality
    print("\n[6/11] Refreshing IL counts…")
    run_il_refresh()

    print("\n[6b/11] Refreshing player WAR…")
    run_player_war_refresh()

    print("\n[6c/11] Refreshing IL quality (WAR-weighted)…")
    run_il_quality_refresh()

    # Step 7: historical odds
    print("\n[7/11] Refreshing historical odds…")
    run_odds_refresh()

    # Step 8: pitcher handedness
    print("\n[8/11] Backfilling pitcher handedness…")
    run_handedness_backfill()

    # Step 9: feature engineering
    if not args.skip_features:
        print("\n[9/11] Rebuilding feature matrix…")
        ok = run_feature_engineering()
        if ok:
            features_path = os.path.join(DATA_DIR, "features.csv")
            if os.path.exists(features_path):
                feat_df = pd.read_csv(features_path)
                print(f"  Features rebuilt  : {len(feat_df):,} rows × {len(feat_df.columns)} cols")
        print_hot_cold_summary()
    else:
        print("\n[9/9] Skipping feature engineering (--skip-features)")

    print(f"\n{'='*45}")
    print("  Update complete.")
    print(f"{'='*45}\n")
