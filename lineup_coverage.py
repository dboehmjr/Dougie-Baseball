"""Report historical lineup coverage against the model feature matrix."""

from __future__ import annotations

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
LINEUPS_PATH = os.path.join(DATA_DIR, "historical_lineups.csv")


def build_lineup_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    lineups = pd.read_csv(LINEUPS_PATH, parse_dates=["game_date"])

    merged = features.merge(
        lineups[["game_date", "home_team", "away_team", "game_pk"]],
        left_on=["Date", "home_team", "away_team"],
        right_on=["game_date", "home_team", "away_team"],
        how="left",
    )

    yearly = (
        merged.groupby("year")
        .agg(
            games=("Date", "count"),
            lineup_rows=("game_pk", lambda s: int(s.notna().sum())),
            home_lineup_vs_sp_coverage=("home_lineup_ops_vs_sp", lambda s: float(s.notna().mean())),
            away_lineup_vs_sp_coverage=("away_lineup_ops_vs_sp", lambda s: float(s.notna().mean())),
            home_sp_hand_coverage=("home_opp_sp_is_lhp", lambda s: float(s.notna().mean())),
            away_sp_hand_coverage=("away_opp_sp_is_lhp", lambda s: float(s.notna().mean())),
        )
        .reset_index()
    )
    yearly["lineup_row_coverage"] = yearly["lineup_rows"] / yearly["games"]

    missing = merged[
        merged["game_pk"].isna()
        | merged["home_lineup_ops_vs_sp"].isna()
        | merged["away_lineup_ops_vs_sp"].isna()
    ][[
        "Date", "year", "home_team", "away_team", "game_pk",
        "home_lineup_ops_vs_sp", "away_lineup_ops_vs_sp",
        "home_opp_sp_is_lhp", "away_opp_sp_is_lhp",
    ]].copy()

    return yearly, missing


def main() -> None:
    yearly, missing = build_lineup_coverage()
    yearly_path = os.path.join(DATA_DIR, "lineup_coverage_by_year.csv")
    missing_path = os.path.join(DATA_DIR, "lineup_coverage_missing_games.csv")
    yearly.to_csv(yearly_path, index=False)
    missing.to_csv(missing_path, index=False)
    print(f"Saved {yearly_path}")
    print(f"Saved {missing_path}")
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
