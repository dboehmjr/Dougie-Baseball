"""
Improve pitcher handedness coverage.

Primary source:
  - Chadwick register via pybaseball.chadwick_register(), matched by Retrosheet
    pitcher IDs from data/game_sp.csv. When the local Chadwick table does not
    include a handedness column, its MLBAM IDs are used for exact MLB Stats API
    person lookups.

Fallback source:
  - MLB Stats API name lookup for unresolved pitchers.

Outputs:
  - data/pitcher_handedness.json  (name_norm -> "R"/"L"/None)
  - data/pitcher_stuff.csv        (Throws filled from cache where possible)

Usage:
  python backfill_handedness.py
  python backfill_handedness.py --refresh-null-api
  python backfill_handedness.py --skip-api
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GAME_SP_PATH = os.path.join(DATA_DIR, "game_sp.csv")
PITCHER_STUFF_PATH = os.path.join(DATA_DIR, "pitcher_stuff.csv")
HAND_CACHE_PATH = os.path.join(DATA_DIR, "pitcher_handedness.json")


def normalize_name(name: str) -> str:
    s = str(name).strip()
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    s = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", s)
    return " ".join(s.replace("-", " ").split())


def _load_cache(path: str = HAND_CACHE_PATH) -> dict[str, str | None]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {normalize_name(k): (v if v in ("R", "L") else None) for k, v in raw.items()}


def _save_cache(cache: dict[str, str | None], path: str = HAND_CACHE_PATH) -> None:
    ordered = {k: cache[k] for k in sorted(cache)}
    with open(path, "w") as f:
        json.dump(ordered, f)


def _valid_hand(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {"R", "L"}:
        return s
    return None


def _name_pairs_from_game_sp(game_sp: pd.DataFrame) -> pd.DataFrame:
    home = game_sp[["home_sp_id", "home_sp_name"]].rename(
        columns={"home_sp_id": "retro_id", "home_sp_name": "name"}
    )
    away = game_sp[["away_sp_id", "away_sp_name"]].rename(
        columns={"away_sp_id": "retro_id", "away_sp_name": "name"}
    )
    pairs = pd.concat([home, away], ignore_index=True).dropna()
    pairs["name_norm"] = pairs["name"].map(normalize_name)
    pairs["retro_id"] = pairs["retro_id"].astype(str).str.strip()
    return pairs.drop_duplicates(["retro_id", "name_norm"])


def _hand_from_person_id(statsapi, person_id: int) -> str | None:
    try:
        person_data = statsapi.get("person", {"personId": int(person_id)})
        person = (person_data.get("people") or [{}])[0]
        return _valid_hand((person.get("pitchHand") or {}).get("code"))
    except Exception:
        return None


def _mlbam_pairs_from_chadwick(pairs: pd.DataFrame, register: pd.DataFrame) -> dict[str, int]:
    if "key_retro" not in register.columns or "key_mlbam" not in register.columns:
        return {}

    reg = register[["key_retro", "key_mlbam"]].copy()
    reg["key_retro"] = reg["key_retro"].astype(str).str.strip()
    reg["key_mlbam"] = pd.to_numeric(reg["key_mlbam"], errors="coerce")
    reg = reg.dropna(subset=["key_retro", "key_mlbam"]).drop_duplicates("key_retro")

    merged = pairs.merge(reg, left_on="retro_id", right_on="key_retro", how="left")
    merged = merged.dropna(subset=["key_mlbam"])
    return {
        row["name_norm"]: int(row["key_mlbam"])
        for _, row in merged.drop_duplicates(["name_norm", "key_mlbam"]).iterrows()
    }


def update_from_mlbam_ids(cache: dict[str, str | None],
                          name_to_mlbam: dict[str, int],
                          refresh_null: bool = False) -> tuple[dict[str, str | None], int, int]:
    """Fill cache through exact MLBAM person IDs."""
    try:
        import statsapi
    except ImportError:
        print("  statsapi not installed; skipping MLBAM ID hydration")
        return cache, 0, 0

    todo = {
        name: pid
        for name, pid in name_to_mlbam.items()
        if name and pid and (name not in cache or (refresh_null and cache.get(name) is None))
    }

    resolved = 0
    print(f"  MLBAM exact-ID handedness lookups: {len(todo):,}")
    for name_norm, person_id in sorted(todo.items()):
        hand = _hand_from_person_id(statsapi, person_id)
        if hand in ("R", "L"):
            resolved += cache.get(name_norm) != hand
            cache[name_norm] = hand
        elif name_norm not in cache:
            cache[name_norm] = None
        time.sleep(0.04)
    return cache, int(resolved), len(todo)


def update_from_chadwick(cache: dict[str, str | None],
                         game_sp_path: str = GAME_SP_PATH,
                         hydrate_mlbam: bool = True,
                         refresh_null: bool = False) -> tuple[dict[str, str | None], int, dict[str, int]]:
    """Fill cache by matching Retrosheet IDs to Chadwick register data."""
    if not os.path.exists(game_sp_path):
        return cache, 0, {}

    import pybaseball as pb

    game_sp = pd.read_csv(game_sp_path)
    pairs = _name_pairs_from_game_sp(game_sp)

    register = pb.chadwick_register(save=True)
    if "key_retro" not in register.columns:
        return cache, 0, {}

    name_to_mlbam = _mlbam_pairs_from_chadwick(pairs, register)
    if "throws" not in register.columns:
        if hydrate_mlbam:
            cache, resolved, _ = update_from_mlbam_ids(
                cache, name_to_mlbam, refresh_null=refresh_null
            )
            return cache, resolved, name_to_mlbam
        return cache, 0, name_to_mlbam

    reg = register[["key_retro", "throws"]].copy()
    reg["key_retro"] = reg["key_retro"].astype(str).str.strip()
    reg["throws"] = reg["throws"].map(_valid_hand)
    reg = reg.dropna(subset=["key_retro", "throws"]).drop_duplicates("key_retro")

    merged = pairs.merge(reg, left_on="retro_id", right_on="key_retro", how="left")
    resolved = 0
    for _, row in merged.dropna(subset=["throws"]).iterrows():
        name_norm = row["name_norm"]
        hand = row["throws"]
        if hand in ("R", "L") and cache.get(name_norm) != hand:
            cache[name_norm] = hand
            resolved += 1
    if hydrate_mlbam:
        cache, id_resolved, _ = update_from_mlbam_ids(
            cache, name_to_mlbam, refresh_null=refresh_null
        )
        resolved += id_resolved
    return cache, resolved, name_to_mlbam


def update_from_statsapi(cache: dict[str, str | None],
                         names: list[str],
                         refresh_null: bool = False) -> tuple[dict[str, str | None], int]:
    """Fallback name lookup through MLB Stats API."""
    try:
        import statsapi
    except ImportError:
        print("  statsapi not installed; skipping MLB API fallback")
        return cache, 0

    to_lookup = []
    for name in names:
        if not name:
            continue
        if name not in cache or (refresh_null and cache.get(name) is None):
            to_lookup.append(name)

    resolved = 0
    print(f"  MLB Stats API fallback lookups: {len(to_lookup)}")
    for name_norm in to_lookup:
        found = None
        try:
            results = statsapi.lookup_player(name_norm)
            for p in results:
                pos = p.get("primaryPosition") or {}
                if pos.get("code") != "1":
                    continue
                pid = p.get("id")
                if not pid:
                    continue
                hand = _hand_from_person_id(statsapi, pid)
                if hand in ("R", "L"):
                    found = hand
                    break
        except Exception:
            found = None

        if found in ("R", "L"):
            resolved += cache.get(name_norm) != found
            cache[name_norm] = found
        elif name_norm not in cache:
            cache[name_norm] = None
        time.sleep(0.08)
    return cache, int(resolved)


def apply_to_pitcher_stuff(cache: dict[str, str | None],
                           pitcher_stuff_path: str = PITCHER_STUFF_PATH) -> tuple[int, int]:
    if not os.path.exists(pitcher_stuff_path):
        return 0, 0
    df = pd.read_csv(pitcher_stuff_path)
    if "name_norm" not in df.columns:
        return 0, len(df)
    if "Throws" not in df.columns:
        df["Throws"] = np.nan

    before = int(df["Throws"].notna().sum())
    df["name_norm"] = df["name_norm"].map(normalize_name)
    df["Throws"] = df.apply(
        lambda r: cache.get(r["name_norm"], r["Throws"])
        if pd.isna(r["Throws"]) or str(r["Throws"]).strip() not in {"R", "L"}
        else r["Throws"],
        axis=1,
    )
    df.to_csv(pitcher_stuff_path, index=False)
    after = int(df["Throws"].notna().sum())
    return after - before, len(df)


def build_name_universe() -> list[str]:
    names: set[str] = set()
    if os.path.exists(PITCHER_STUFF_PATH):
        stuff = pd.read_csv(PITCHER_STUFF_PATH)
        if "name_norm" in stuff.columns:
            names.update(stuff["name_norm"].dropna().map(normalize_name))
    if os.path.exists(GAME_SP_PATH):
        sp = pd.read_csv(GAME_SP_PATH)
        for col in ["home_sp_name", "away_sp_name"]:
            if col in sp.columns:
                names.update(sp[col].dropna().map(normalize_name))
    return sorted(n for n in names if n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pitcher handedness")
    parser.add_argument("--skip-api", action="store_true", help="Only use Chadwick/register/local sources")
    parser.add_argument("--refresh-null-api", action="store_true", help="Retry names cached as null via MLB Stats API")
    args = parser.parse_args()

    cache = _load_cache()
    initial_resolved = sum(v in ("R", "L") for v in cache.values())
    print(f"Initial cache: {len(cache):,} names, {initial_resolved:,} resolved")

    try:
        cache, chadwick_resolved, name_to_mlbam = update_from_chadwick(
            cache,
            hydrate_mlbam=not args.skip_api,
            refresh_null=args.refresh_null_api,
        )
        print(f"Chadwick/MLBAM resolved/updated: {chadwick_resolved:,}")
        print(f"Chadwick MLBAM ID matches: {len(name_to_mlbam):,}")
    except Exception as exc:
        print(f"Chadwick lookup failed: {exc}")
        chadwick_resolved = 0

    if not args.skip_api:
        names = build_name_universe()
        cache, api_resolved = update_from_statsapi(
            cache, names, refresh_null=args.refresh_null_api
        )
        print(f"MLB API resolved/updated: {api_resolved:,}")

    _save_cache(cache)
    stuff_added, stuff_total = apply_to_pitcher_stuff(cache)
    final_resolved = sum(v in ("R", "L") for v in cache.values())
    print(f"Final cache: {len(cache):,} names, {final_resolved:,} resolved")
    print(f"pitcher_stuff.csv Throws added: {stuff_added:,}; rows: {stuff_total:,}")


if __name__ == "__main__":
    main()
