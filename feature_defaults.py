"""Neutral defaults for model feature completeness.

These values are intentionally conservative: they represent "no known edge"
rather than optimistic estimates.  The goal is to keep DB feature rows complete
without manufacturing directional signal.
"""

from __future__ import annotations

import pandas as pd


NEUTRAL_DEFAULTS: dict[str, float] = {
    "h2h_home_win_rate": 0.5,
    "h2h_home_run_diff": 0.0,
    "home_prior_win_pct": 0.5,
    "away_prior_win_pct": 0.5,
    "home_season_win_pct": 0.5,
    "away_season_win_pct": 0.5,
    "home_season_win_pct_delta": 0.0,
    "away_season_win_pct_delta": 0.0,
    "temp_f": 70.0,
    "wind_speed_mph": 0.0,
    "wind_to_cf": 0.0,
    "humidity_pct": 55.0,
    "ump_run_factor": 1.0,
    "home_il_count": 0.0,
    "away_il_count": 0.0,
    "home_il_war": 0.0,
    "away_il_war": 0.0,
    "home_sp_era_adj": 4.3,
    "away_sp_era_adj": 4.3,
    "home_sp_inseason_era": 4.3,
    "away_sp_inseason_era": 4.3,
    "home_sp_last3_era": 4.3,
    "away_sp_last3_era": 4.3,
    "home_bullpen_inseason_era": 4.3,
    "away_bullpen_inseason_era": 4.3,
    "home_offense_re24_15g": 0.0,
    "away_offense_re24_15g": 0.0,
    "home_sp_re24_last3": 0.0,
    "away_sp_re24_last3": 0.0,
    "home_bullpen_re24_15d": 0.0,
    "away_bullpen_re24_15d": 0.0,
    "home_sp_fbv": 93.0,
    "away_sp_fbv": 93.0,
    "home_sp_swstr": 0.11,
    "away_sp_swstr": 0.11,
    "home_sp_k_pct": 0.22,
    "away_sp_k_pct": 0.22,
    "home_sp_xfip": 4.3,
    "away_sp_xfip": 4.3,
    "home_sp_pa": 0.0,
    "away_sp_pa": 0.0,
    "home_sp_days_rest": 5.0,
    "away_sp_days_rest": 5.0,
    "home_sp_outs_last": 15.0,
    "away_sp_outs_last": 15.0,
    "home_bullpen_outs_3d": 0.0,
    "away_bullpen_outs_3d": 0.0,
    "home_team_ops": 0.720,
    "away_team_ops": 0.720,
    "home_batting_ops_vs_sp": 0.720,
    "away_batting_ops_vs_sp": 0.720,
    "home_platoon_advantage": 0.0,
    "away_platoon_advantage": 0.0,
    "home_lineup_ops_vs_sp": 0.720,
    "away_lineup_ops_vs_sp": 0.720,
    "home_lineup_ops_vs_lhp": 0.720,
    "home_lineup_ops_vs_rhp": 0.720,
    "away_lineup_ops_vs_lhp": 0.720,
    "away_lineup_ops_vs_rhp": 0.720,
    "home_lineup_known_batters": 0.0,
    "away_lineup_known_batters": 0.0,
    "home_barrel_pct": 7.0,
    "away_barrel_pct": 7.0,
    "home_hard_hit_pct": 38.0,
    "away_hard_hit_pct": 38.0,
    "home_home_rd": 0.0,
    "home_away_rd": 0.0,
    "away_home_rd": 0.0,
    "away_away_rd": 0.0,
    "home_rd_split": 0.0,
    "away_rd_split": 0.0,
    "rd_venue_diff": 0.0,
    "away_travel_miles": 0.0,
    "travel_diff": 0.0,
    "vegas_home_prob": 0.5,
    "park_factor": 1.0,
    "home_opp_sp_is_lhp": 0.0,
    "away_opp_sp_is_lhp": 0.0,
    "home_sp_is_lhp": 0.0,
    "away_sp_is_lhp": 0.0,
    "both_sp_same_hand": 0.0,
}


DIFF_PAIRS: dict[str, tuple[str, str, int]] = {
    "prior_win_pct_diff": ("home_prior_win_pct", "away_prior_win_pct", 1),
    "season_win_pct_diff": ("home_season_win_pct", "away_season_win_pct", 1),
    "season_win_pct_delta_diff": ("home_season_win_pct_delta", "away_season_win_pct_delta", 1),
    "il_diff": ("home_il_count", "away_il_count", 1),
    "il_war_diff": ("home_il_war", "away_il_war", 1),
    "sp_era_adj_diff": ("away_sp_era_adj", "home_sp_era_adj", 1),
    "sp_inseason_era_diff": ("away_sp_inseason_era", "home_sp_inseason_era", 1),
    "sp_last3_era_diff": ("away_sp_last3_era", "home_sp_last3_era", 1),
    "bullpen_inseason_era_diff": ("away_bullpen_inseason_era", "home_bullpen_inseason_era", 1),
    "offense_re24_diff": ("home_offense_re24_15g", "away_offense_re24_15g", 1),
    "sp_re24_diff": ("home_sp_re24_last3", "away_sp_re24_last3", 1),
    "bullpen_re24_diff": ("home_bullpen_re24_15d", "away_bullpen_re24_15d", 1),
    "sp_fbv_diff": ("home_sp_fbv", "away_sp_fbv", 1),
    "sp_swstr_diff": ("home_sp_swstr", "away_sp_swstr", 1),
    "sp_k_pct_diff": ("home_sp_k_pct", "away_sp_k_pct", 1),
    "sp_xfip_diff": ("away_sp_xfip", "home_sp_xfip", 1),
    "sp_pa_diff": ("home_sp_pa", "away_sp_pa", 1),
    "sp_days_rest_diff": ("home_sp_days_rest", "away_sp_days_rest", 1),
    "sp_outs_last_diff": ("home_sp_outs_last", "away_sp_outs_last", 1),
    "bullpen_usage_diff": ("away_bullpen_outs_3d", "home_bullpen_outs_3d", 1),
    "ops_diff": ("home_team_ops", "away_team_ops", 1),
    "batting_ops_vs_sp_diff": ("home_batting_ops_vs_sp", "away_batting_ops_vs_sp", 1),
    "platoon_advantage_diff": ("home_platoon_advantage", "away_platoon_advantage", 1),
    "lineup_ops_vs_sp_diff": ("home_lineup_ops_vs_sp", "away_lineup_ops_vs_sp", 1),
    "lineup_ops_vs_lhp_diff": ("home_lineup_ops_vs_lhp", "away_lineup_ops_vs_lhp", 1),
    "lineup_ops_vs_rhp_diff": ("home_lineup_ops_vs_rhp", "away_lineup_ops_vs_rhp", 1),
    "lineup_known_batters_diff": ("home_lineup_known_batters", "away_lineup_known_batters", 1),
    "barrel_pct_diff": ("home_barrel_pct", "away_barrel_pct", 1),
    "hard_hit_pct_diff": ("home_hard_hit_pct", "away_hard_hit_pct", 1),
}


def apply_feature_defaults(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Fill known model features with neutral values and recompute safe diffs."""
    out = df.copy()
    target_cols = set(feature_cols or out.columns)

    for col, default in NEUTRAL_DEFAULTS.items():
        if col in out.columns and col in target_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    for col in list(target_cols):
        if col in out.columns and (
            col.endswith("_diff")
            or col.endswith("_delta")
            or col.endswith("_advantage")
            or col.endswith("_momentum")
        ):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for out_col, (left, right, sign) in DIFF_PAIRS.items():
        if out_col in out.columns and out_col in target_cols and left in out.columns and right in out.columns:
            out[out_col] = (
                pd.to_numeric(out[left], errors="coerce")
                - pd.to_numeric(out[right], errors="coerce")
            ) * sign
            out[out_col] = out[out_col].fillna(0.0)

    return out
