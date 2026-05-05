"""Shared moneyline betting policy.

The baseball model estimates win probability. This layer decides whether the
model-vs-market edge is actionable after accounting for where past edge has
actually held up by moneyline range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import os

import numpy as np
import pandas as pd


DEFAULT_MIN_EDGE = 0.06
DEFAULT_ALLOWED_BUCKET = "+130:+170"


@dataclass(frozen=True)
class OddsBucketRule:
    bucket: str
    min_odds: float
    max_odds: float
    min_edge: float | None
    note: str

    def matches(self, odds: float) -> bool:
        return self.min_odds <= odds < self.max_odds


ODDS_BUCKET_RULES = [
    OddsBucketRule("<-180", -np.inf, -180, None, "deep favorite bucket has been unprofitable"),
    OddsBucketRule("-180:-140", -180, -140, None, "favorite bucket failed holdout/live check"),
    OddsBucketRule("-140:-115", -140, -115, None, "favorite bucket has been unprofitable"),
    OddsBucketRule("-115:+100", -115, 100, None, "near pick'em bucket has been unprofitable"),
    OddsBucketRule("+100:+130", 100, 130, None, "short underdog bucket has been unprofitable"),
    OddsBucketRule("+130:+170", 130, 170, 0.06, "allowed underdog bucket"),
    OddsBucketRule("+170+", 170, np.inf, None, "long underdog bucket has been unprofitable"),
]


def odds_bucket(odds: float | None) -> str | None:
    if odds is None or pd.isna(odds):
        return None
    for rule in ODDS_BUCKET_RULES:
        if rule.matches(float(odds)):
            return rule.bucket
    return None


def rule_for_odds(odds: float | None) -> OddsBucketRule | None:
    if odds is None or pd.isna(odds):
        return None
    for rule in ODDS_BUCKET_RULES:
        if rule.matches(float(odds)):
            return rule
    return None


@dataclass(frozen=True)
class PolicyRule:
    bucket: str
    min_edge: float
    season_phase: str = "all"
    min_model_prob: float = 0.0


@dataclass(frozen=True)
class BettingPolicy:
    rules: tuple[PolicyRule, ...]
    name: str = "static"

    def threshold_for(self, bucket: str | None, season_phase: str | None = None) -> float | None:
        rule = self.rule_for(bucket, season_phase)
        return rule.min_edge if rule else None

    def rule_for(self, bucket: str | None, season_phase: str | None = None) -> PolicyRule | None:
        if bucket is None:
            return None
        phase = season_phase or "all"
        for rule in self.rules:
            if rule.bucket == bucket and rule.season_phase == phase:
                return rule
        for rule in self.rules:
            if rule.bucket == bucket and rule.season_phase == "all":
                return rule
        return None


DEFAULT_POLICY = BettingPolicy(
    rules=(PolicyRule(DEFAULT_ALLOWED_BUCKET, DEFAULT_MIN_EDGE),),
    name="static_underdog_130_170",
)


def policy_to_dict(policy: BettingPolicy) -> dict:
    return {
        "name": policy.name,
        "rules": [
            {
                "bucket": rule.bucket,
                "min_edge": rule.min_edge,
                "season_phase": rule.season_phase,
                "min_model_prob": rule.min_model_prob,
            }
            for rule in policy.rules
        ],
    }


def policy_from_dict(data: dict) -> BettingPolicy:
    return BettingPolicy(
        rules=tuple(
            PolicyRule(
                bucket=str(rule["bucket"]),
                min_edge=float(rule["min_edge"]),
                season_phase=str(rule.get("season_phase", "all")),
                min_model_prob=float(rule.get("min_model_prob", 0.0)),
            )
            for rule in data.get("rules", [])
        ),
        name=str(data.get("name", "loaded_policy")),
    )


def save_policy(policy: BettingPolicy, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy_to_dict(policy), f, indent=2)


def load_policy(path: str, default: BettingPolicy = DEFAULT_POLICY) -> BettingPolicy:
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return policy_from_dict(json.load(f))


def season_phase(game_date: str | date | datetime | pd.Timestamp | None) -> str:
    """Coarse season phase for policy tuning/calibration."""
    if game_date is None or pd.isna(game_date):
        return "unknown"
    dt = pd.to_datetime(game_date)
    if dt.month <= 4:
        return "early"
    if dt.month >= 9:
        return "late"
    return "mid"


def evaluate_candidate(
    edge: float,
    odds: float | None,
    *,
    model_prob: float | None = None,
    policy: BettingPolicy | None = None,
    game_date: str | date | datetime | pd.Timestamp | None = None,
) -> tuple[bool, float | None, str | None, str]:
    """Return whether a candidate side passes the odds-aware policy."""
    active_policy = policy or DEFAULT_POLICY
    rule = rule_for_odds(odds)
    if rule is None:
        return False, None, None, "missing odds"
    phase = season_phase(game_date)
    policy_rule = active_policy.rule_for(rule.bucket, phase)
    if policy_rule is None:
        return False, None, rule.bucket, f"{rule.bucket} excluded by {active_policy.name}"
    min_edge = policy_rule.min_edge
    if pd.isna(edge):
        return False, min_edge, rule.bucket, "missing edge"
    if model_prob is not None and not pd.isna(model_prob) and model_prob < policy_rule.min_model_prob:
        return False, min_edge, rule.bucket, f"model prob < {policy_rule.min_model_prob:.0%}"
    if edge > min_edge:
        return True, min_edge, rule.bucket, f"{active_policy.name}: {rule.bucket} edge > {min_edge:.0%}"
    return False, min_edge, rule.bucket, f"edge <= {min_edge:.0%} threshold"


def choose_bet(
    *,
    home: str,
    away: str,
    p_home: float,
    home_ml: float,
    away_ml: float,
    vegas_home: float,
    policy: BettingPolicy | None = None,
    game_date: str | date | datetime | pd.Timestamp | None = None,
) -> dict:
    """Pick the larger raw edge, then apply odds-bucket actionability rules."""
    p_away = 1 - p_home
    vegas_away = 1 - vegas_home
    edge_home = p_home - vegas_home
    edge_away = p_away - vegas_away

    if edge_home >= edge_away:
        side = home
        model_prob = p_home
        vegas_prob = vegas_home
        odds = home_ml
        edge = edge_home
    else:
        side = away
        model_prob = p_away
        vegas_prob = vegas_away
        odds = away_ml
        edge = edge_away

    phase = season_phase(game_date)
    should_bet, min_edge, bucket, reason = evaluate_candidate(
        edge, odds, model_prob=model_prob, policy=policy, game_date=game_date
    )
    return {
        "side": side,
        "model_prob": model_prob,
        "vegas_prob": vegas_prob,
        "odds": odds,
        "edge": edge,
        "odds_bucket": bucket,
        "season_phase": phase,
        "edge_threshold": min_edge,
        "policy_reason": reason,
        "should_bet": should_bet,
    }
