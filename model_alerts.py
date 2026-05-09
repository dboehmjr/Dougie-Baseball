from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PYBASEBALL_CACHE", str(BASE_DIR / "logs" / "pybaseball_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "logs" / "matplotlib_cache"))
(BASE_DIR / "logs" / "pybaseball_cache").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "logs" / "matplotlib_cache").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import requests
import zoneinfo

from betting_strategy import choose_bet, load_policy
from fetch_lineups import fetch_confirmed_lineups
from fetch_odds import load_or_fetch_odds
from predict import predict_matchup


DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ALERT_LOG_PATH = DATA_DIR / "model_alerts_log.csv"
POLICY_PATH = MODELS_DIR / "betting_policy.json"
LEDGER_PATH = DATA_DIR / "bankroll_ledger.csv"
ENV_PATH = BASE_DIR / ".env"

ET = zoneinfo.ZoneInfo("America/New_York")
MLB_API_MAP = {
    "WSH": "WSN", "SD": "SDP", "TB": "TBR",
    "KC": "KCR", "AZ": "ARI", "SF": "SFG",
    "OAK": "ATH", "ATH": "ATH", "CWS": "CHW",
}

ALERT_COLS = [
    "sent_at", "game_date", "alert_type", "home", "away", "side",
    "stake", "odds", "model_prob", "vegas_prob", "edge",
    "lineup_status", "home_sp", "away_sp", "game_time",
]


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def fetch_schedule(game_date: str) -> list[dict]:
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date}&hydrate=team,linescore,probablePitcher"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for g in data.get("dates", [{}])[0].get("games", []):
        home_raw = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
        away_raw = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
        home = MLB_API_MAP.get(home_raw, home_raw)
        away = MLB_API_MAP.get(away_raw, away_raw)
        status = g.get("status", {}).get("detailedState", "")
        game_date_utc = g.get("gameDate", "")
        try:
            start_dt = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
            start_et = start_dt.astimezone(ET)
            game_time = start_et.strftime("%-I:%M %p")
        except Exception:
            start_et = None
            game_time = "TBD"
        out.append({
            "game_pk": g.get("gamePk"),
            "home": home,
            "away": away,
            "status": status,
            "home_sp": g.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD"),
            "away_sp": g.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD"),
            "start_et": start_et,
            "game_time": game_time,
        })
    return out


def implied_prob(ml: float) -> float:
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def kelly_stake(p: float, bankroll: float, frac: float, ml: float, max_pct: float = 0.05) -> float:
    net = (ml / 100) if ml > 0 else (100 / abs(ml))
    edge = p * net - (1 - p)
    if edge <= 0:
        return 0.0
    k = frac * (edge / net)
    return round(min(k, max_pct) * bankroll, 2)


def load_bankroll(default: float = 100.0) -> float:
    if not LEDGER_PATH.exists():
        return default
    try:
        ledger = pd.read_csv(LEDGER_PATH)
        if not ledger.empty and "bankroll" in ledger.columns:
            val = pd.to_numeric(ledger["bankroll"], errors="coerce").dropna()
            if not val.empty:
                return float(val.iloc[-1])
    except Exception:
        pass
    return default


def load_alert_log() -> pd.DataFrame:
    if ALERT_LOG_PATH.exists():
        return pd.read_csv(ALERT_LOG_PATH)
    return pd.DataFrame(columns=ALERT_COLS)


def save_alert_log(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALERT_LOG_PATH, index=False)


def already_sent(log: pd.DataFrame, game_date: str, alert_type: str, home: str, away: str) -> bool:
    if log.empty:
        return False
    mask = (
        (log["game_date"].astype(str) == game_date)
        & (log["alert_type"] == alert_type)
        & (log["home"] == home)
        & (log["away"] == away)
    )
    return bool(mask.any())


def send_pushover(title: str, message: str, dry_run: bool = False) -> bool:
    user_key = os.environ.get("PUSHOVER_USER_KEY")
    api_token = os.environ.get("PUSHOVER_API_TOKEN")
    if dry_run or not user_key or not api_token:
        print(f"\n{title}\n{message}\n")
        return False
    resp = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": api_token, "user": user_key, "title": title, "message": message},
        timeout=15,
    )
    resp.raise_for_status()
    return True


def lineup_status(home: str, away: str, lineups: pd.DataFrame) -> tuple[str, bool]:
    if lineups is None or lineups.empty:
        return "pending", False
    row = lineups[(lineups["home_team"] == home) & (lineups["away_team"] == away)]
    if row.empty:
        return "pending", False
    h_conf = bool(row.iloc[0].get("home_lineup_confirmed", False))
    a_conf = bool(row.iloc[0].get("away_lineup_confirmed", False))
    if h_conf and a_conf:
        return "both confirmed", True
    if h_conf or a_conf:
        return "one confirmed", False
    return "pending", False


def format_odds(odds: float) -> str:
    odds = int(round(float(odds)))
    return f"+{odds}" if odds > 0 else str(odds)


def alert_message(alert_type: str, game: dict, pick: dict, stake: float,
                  lineup_label: str) -> tuple[str, str]:
    title = "MLB Model Alert" if alert_type == "final_lineup" else "MLB Early Alert"
    header = (
        f"{game['away']} @ {game['home']} is ready"
        if alert_type == "final_lineup"
        else f"{game['away']} @ {game['home']}"
    )
    message = "\n".join([
        header,
        "",
        f"Suggested Side: {pick['side']}",
        f"Stake: ${stake:.2f}",
        f"Line: {pick['side']} {format_odds(pick['odds'])}",
        f"Win Prob: {pick['model_prob']:.1%}",
        f"Vegas Prob: {pick['vegas_prob']:.1%}",
        f"Edge: {pick['edge']:+.1%}",
        f"Lineups: {lineup_label}",
        "SPs: confirmed",
        f"Starts: {game['game_time']}",
    ])
    return title, message


def evaluate_alerts(game_date: str, dry_run: bool = False) -> int:
    load_dotenv()
    now_et = datetime.now(ET)
    active_start = int(os.environ.get("MODEL_ALERT_ACTIVE_START_HOUR", "10"))
    active_end = int(os.environ.get("MODEL_ALERT_ACTIVE_END_HOUR", "23"))
    if not dry_run and not (active_start <= now_et.hour < active_end):
        print(
            f"Outside active alert window "
            f"({active_start:02d}:00-{active_end:02d}:00 ET); skipping"
        )
        return 0

    policy = load_policy(str(POLICY_PATH))
    bankroll = load_bankroll()
    kelly_frac = float(os.environ.get("MODEL_ALERT_KELLY_FRAC", "0.25"))
    min_stake = float(os.environ.get("MODEL_ALERT_MIN_STAKE", "0.01"))
    min_minutes_to_start = int(os.environ.get("MODEL_ALERT_MIN_MINUTES_TO_START", "5"))

    schedule = fetch_schedule(game_date)
    if not schedule:
        print(f"No games found for {game_date}")
        return 0

    odds_df = load_or_fetch_odds(game_date=game_date, expected_games=len(schedule))
    lineups = fetch_confirmed_lineups(game_date=game_date)
    log = load_alert_log()
    sent_rows = []

    for game in schedule:
        home, away = game["home"], game["away"]
        status = game["status"]
        if "Final" in status or "Progress" in status or "Delayed" in status:
            continue
        if game["start_et"] is not None:
            minutes_to_start = (game["start_et"] - now_et).total_seconds() / 60
            if minutes_to_start < min_minutes_to_start:
                continue
        if game["home_sp"] in ("", "TBD") or game["away_sp"] in ("", "TBD"):
            continue

        orow = odds_df[(odds_df["home_team"] == home) & (odds_df["away_team"] == away)]
        if orow.empty:
            continue
        home_ml = float(orow.iloc[0].get("home_ml", np.nan))
        away_ml = float(orow.iloc[0].get("away_ml", np.nan))
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        imp_sum = implied_prob(home_ml) + implied_prob(away_ml)
        if imp_sum < 1.0:
            continue
        vegas_home = implied_prob(home_ml) / imp_sum

        try:
            p_home = predict_matchup(
                home, away, int(game_date[:4]), game_date=game_date,
                home_sp_name=game["home_sp"], away_sp_name=game["away_sp"],
                verbose=False,
            )
        except Exception as exc:
            print(f"Prediction failed for {away}@{home}: {exc}")
            continue

        pick = choose_bet(
            home=home, away=away, p_home=p_home,
            home_ml=home_ml, away_ml=away_ml, vegas_home=vegas_home,
            policy=policy, game_date=game_date,
        )
        if not pick["should_bet"]:
            continue
        stake = kelly_stake(pick["model_prob"], bankroll, kelly_frac, pick["odds"])
        if stake < min_stake:
            continue

        lineup_label, both_lineups = lineup_status(home, away, lineups)
        alert_types = ["early_actionable"]
        if both_lineups:
            alert_types.append("final_lineup")

        for alert_type in alert_types:
            if already_sent(log, game_date, alert_type, home, away):
                continue
            title, message = alert_message(alert_type, game, pick, stake, lineup_label)
            delivered = send_pushover(title, message, dry_run=dry_run)
            if delivered:
                sent_rows.append({
                    "sent_at": datetime.now(ET).isoformat(timespec="seconds"),
                    "game_date": game_date,
                    "alert_type": alert_type,
                    "home": home,
                    "away": away,
                    "side": pick["side"],
                    "stake": stake,
                    "odds": pick["odds"],
                    "model_prob": pick["model_prob"],
                    "vegas_prob": pick["vegas_prob"],
                    "edge": pick["edge"],
                    "lineup_status": lineup_label,
                    "home_sp": game["home_sp"],
                    "away_sp": game["away_sp"],
                    "game_time": game["game_time"],
                })
            time.sleep(0.3)

    if sent_rows and not dry_run:
        log = pd.concat([log, pd.DataFrame(sent_rows)], ignore_index=True)
        save_alert_log(log)
    print(f"Sent {len(sent_rows)} alert(s) for {game_date}")
    return len(sent_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate_alerts(args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
