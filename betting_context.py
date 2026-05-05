"""
Shared betting-context helpers for backtests and diagnostics.

The current odds caches only contain one consensus moneyline snapshot. These
helpers also understand optional closing-line columns, so CLV appears
automatically once the source data includes close prices/probabilities.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GAME_SP_PATH = os.path.join(DATA_DIR, "game_sp.csv")
PITCHER_HANDEDNESS_PATH = os.path.join(DATA_DIR, "pitcher_handedness.json")


def american_to_implied(ml: float) -> float:
    if pd.isna(ml):
        return np.nan
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def devig_prob(home_ml: float, away_ml: float) -> float:
    ph = american_to_implied(home_ml)
    pa = american_to_implied(away_ml)
    total = ph + pa
    return ph / total if total and total > 0 else np.nan


def normalize_pitcher_name(name: str) -> str:
    s = str(name).strip()
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    s = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", s)
    return " ".join(s.replace("-", " ").split())


def load_pitcher_handedness(path: str = PITCHER_HANDEDNESS_PATH) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {normalize_pitcher_name(k): v for k, v in data.items()}


def attach_starting_pitcher_context(df: pd.DataFrame,
                                    game_sp_path: str = GAME_SP_PATH,
                                    handedness_path: str = PITCHER_HANDEDNESS_PATH) -> pd.DataFrame:
    """Attach SP names and handedness to a game-level DataFrame when available."""
    if not os.path.exists(game_sp_path):
        return df

    out = df.copy()
    sp = pd.read_csv(game_sp_path, parse_dates=["Date"])
    sp["game_date"] = sp["Date"].dt.strftime("%Y-%m-%d")
    keep = ["game_date", "home_team", "away_team", "home_sp_name", "away_sp_name"]
    sp = sp[keep].drop_duplicates(["game_date", "home_team", "away_team"], keep="first")
    out = out.merge(sp, on=["game_date", "home_team", "away_team"], how="left")

    handed = load_pitcher_handedness(handedness_path)
    out["home_sp_norm"] = out["home_sp_name"].map(
        lambda n: normalize_pitcher_name(n) if pd.notna(n) else np.nan
    )
    out["away_sp_norm"] = out["away_sp_name"].map(
        lambda n: normalize_pitcher_name(n) if pd.notna(n) else np.nan
    )
    out["home_sp_throws"] = out["home_sp_norm"].map(handed)
    out["away_sp_throws"] = out["away_sp_norm"].map(handed)
    return out


def odds_merge_columns(odds_df: pd.DataFrame) -> list[str]:
    """Return standard + optional odds columns available for game merges."""
    desired = [
        "game_date", "home_team", "away_team",
        "home_ml", "away_ml", "consensus_prob",
        "open_home_ml", "open_away_ml", "open_consensus_prob",
        "close_home_ml", "close_away_ml", "close_consensus_prob",
        "closing_home_ml", "closing_away_ml", "closing_consensus_prob",
        "home_close_ml", "away_close_ml",
    ]
    return [c for c in desired if c in odds_df.columns]


def _first_present(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return np.nan


def get_close_home_prob(row: pd.Series) -> float:
    """Return devigged closing home probability if closing fields exist."""
    direct = _first_present(row, [
        "close_consensus_prob",
        "closing_consensus_prob",
        "close_home_prob",
        "closing_home_prob",
    ])
    if pd.notna(direct):
        return direct

    close_home_ml = _first_present(row, ["close_home_ml", "closing_home_ml", "home_close_ml"])
    close_away_ml = _first_present(row, ["close_away_ml", "closing_away_ml", "away_close_ml"])
    if pd.notna(close_home_ml) and pd.notna(close_away_ml):
        return devig_prob(close_home_ml, close_away_ml)
    return np.nan


def enrich_bet_record(base: dict,
                      *,
                      row: pd.Series,
                      side: str | None,
                      model_prob: float | None,
                      vegas_prob: float | None,
                      odds: float | None) -> dict:
    """Add context fields to a ledger row."""
    home = row.get("home_team")
    away = row.get("away_team")
    is_home_side = side == home if side is not None else None
    is_away_side = side == away if side is not None else None

    close_home_prob = get_close_home_prob(row)
    if pd.notna(close_home_prob) and side is not None:
        close_prob = close_home_prob if is_home_side else 1 - close_home_prob
        clv_prob = close_prob - vegas_prob if vegas_prob is not None and pd.notna(vegas_prob) else np.nan
    else:
        close_prob = np.nan
        clv_prob = np.nan

    if is_home_side:
        sp_name = row.get("home_sp_name")
        opp_sp_name = row.get("away_sp_name")
        sp_throws = row.get("home_sp_throws")
        opp_sp_throws = row.get("away_sp_throws")
    elif is_away_side:
        sp_name = row.get("away_sp_name")
        opp_sp_name = row.get("home_sp_name")
        sp_throws = row.get("away_sp_throws")
        opp_sp_throws = row.get("home_sp_throws")
    else:
        sp_name = opp_sp_name = sp_throws = opp_sp_throws = np.nan

    base.update({
        "home_team": home,
        "away_team": away,
        "side_home_away": "home" if is_home_side else "away" if is_away_side else None,
        "is_favorite": bool(odds < 0) if odds is not None and pd.notna(odds) else None,
        "sp_name": sp_name,
        "opp_sp_name": opp_sp_name,
        "sp_throws": sp_throws,
        "opp_sp_throws": opp_sp_throws,
        "close_prob": round(float(close_prob), 4) if pd.notna(close_prob) else np.nan,
        "clv_prob": round(float(clv_prob), 4) if pd.notna(clv_prob) else np.nan,
    })
    return base
