"""
Fetch SP pitch velocity, swing-and-miss rate, K%, xERA, and handedness
from Baseball Savant (Statcast) via pybaseball — no FanGraphs scraping required.

Sources:
  - statcast_pitcher_pitch_arsenal     → FBv (ff_avg_speed)
  - statcast_pitcher_arsenal_stats     → whiff_percent, k_percent (by pitch type; weighted avg)
  - statcast_pitcher_expected_stats    → xera (xFIP proxy), pa
  - pitching_stats_bref                → GS, handedness (Throws), team

Output: data/pitcher_stuff.csv
  name_norm, team, year, Throws, FBv, SwStr_pct, K_pct, xFIP

Incremental: only fetches years beyond the cached max (always re-fetches current year).
"""

from __future__ import annotations

import os
import time
import unicodedata
import re
import warnings
import numpy as np
import pandas as pd
import pybaseball as pb

warnings.filterwarnings("ignore")
pb.cache.enable()

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "pitcher_stuff.csv")

# MLB Stats API team name → our abbreviation (for BRef team names)
BREF_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    # BRef short forms
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    "SD": "SDP", "SF": "SFG", "TB": "TBR", "KC": "KCR",
    "WSH": "WSN", "CWS": "CHW", "AZ": "ARI",
}


def _normalize_name(name: str) -> str:
    name = str(name).strip()
    nfkd = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    name = name.encode("ascii", "ignore").decode("ascii").lower()
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    return " ".join(name.split())


def _fetch_year(year: int) -> pd.DataFrame:
    """Fetch and merge all Statcast + BRef pitcher data for one year."""
    rows: dict[int, dict] = {}  # keyed by MLB player_id

    # 1. Fastball velocity (ff_avg_speed = 4-seam; fall back to sinker si_avg_speed)
    try:
        velo = pb.statcast_pitcher_pitch_arsenal(year)
        for _, r in velo.iterrows():
            pid = int(r["pitcher"])
            fbv = r.get("ff_avg_speed")
            if pd.isna(fbv):
                fbv = r.get("si_avg_speed", np.nan)
            rows.setdefault(pid, {})["FBv"] = float(fbv) if pd.notna(fbv) else np.nan
            rows[pid]["name_raw"] = str(r.get("last_name, first_name", ""))
        time.sleep(0.5)
    except Exception as e:
        print(f"    FBv fetch failed: {e}")

    # 2. Whiff% and K% — arsenal_stats is per pitch type; weight by pitch count
    # Also grab team_name_alt (3-letter codes) as primary team source
    try:
        ars = pb.statcast_pitcher_arsenal_stats(year)
        for pid_val, grp in ars.groupby("player_id"):
            pid = int(pid_val)
            rows.setdefault(pid, {})
            total = grp["pitches"].sum() if "pitches" in grp else len(grp)
            if total == 0:
                continue
            if "pitches" in grp.columns:
                weights = grp["pitches"] / total
            else:
                weights = pd.Series([1 / len(grp)] * len(grp), index=grp.index)
            for col, dest in [("whiff_percent", "SwStr_pct"), ("k_percent", "K_pct")]:
                if col in grp.columns:
                    val = (grp[col] * weights).sum()
                    rows[pid][dest] = float(val) / 100 if val > 1 else float(val)
            if "last_name, first_name" in grp.columns:
                rows[pid]["name_raw"] = str(grp["last_name, first_name"].iloc[0])
            # team_name_alt contains clean 3-letter codes (e.g. "LAD", "MIA")
            if "team_name_alt" in grp.columns:
                tm_raw = str(grp["team_name_alt"].iloc[0]).strip().upper()
                rows[pid]["team_statcast"] = BREF_TEAM_MAP.get(tm_raw, tm_raw)
        time.sleep(0.5)
    except Exception as e:
        print(f"    WhiffK fetch failed: {e}")

    # 3. xERA (xFIP proxy) and PA from expected stats
    try:
        exp = pb.statcast_pitcher_expected_stats(year)
        for _, r in exp.iterrows():
            pid = int(r["player_id"])
            rows.setdefault(pid, {})
            rows[pid]["xFIP"] = float(r["xera"]) if pd.notna(r.get("xera")) else np.nan
            rows[pid]["pa"]   = int(r["pa"])      if pd.notna(r.get("pa"))   else 0
            if "last_name, first_name" in r and pd.notna(r["last_name, first_name"]):
                rows[pid]["name_raw"] = str(r["last_name, first_name"])
        time.sleep(0.5)
    except Exception as e:
        print(f"    xERA fetch failed: {e}")

    # 4. GS and handedness from Baseball Reference
    bref_throws: dict[str, str] = {}   # name_norm → Throws
    bref_team:   dict[str, str] = {}   # name_norm → team abbrev
    bref_gs:     dict[str, int] = {}
    try:
        bref = pb.pitching_stats_bref(year)
        # Filter to MLB starters with meaningful innings
        bref = bref[bref["Lev"].str.contains("Maj", na=False)]
        bref = bref[pd.to_numeric(bref["GS"], errors="coerce").fillna(0) >= 1]
        for _, r in bref.iterrows():
            nn = _normalize_name(r.get("Name", ""))
            # Handedness not in BRef — skip; will be NaN
            tm_raw = str(r.get("Tm", "")).strip()
            bref_team[nn] = BREF_TEAM_MAP.get(tm_raw, tm_raw.upper())
            bref_gs[nn]   = int(pd.to_numeric(r.get("GS", 0), errors="coerce") or 0)
        time.sleep(0.5)
    except Exception as e:
        print(f"    BRef fetch failed: {e}")

    if not rows:
        return pd.DataFrame()

    records = []
    for pid, data in rows.items():
        name_raw  = data.get("name_raw", "")
        name_norm = _normalize_name(name_raw)
        gs        = bref_gs.get(name_norm, 0)
        team      = data.get("team_statcast") or bref_team.get(name_norm, "")
        records.append({
            "name_norm": name_norm,
            "team":      team,
            "year":      year,
            "Throws":    None,   # BRef doesn't expose handedness; stays NaN
            "FBv":       data.get("FBv",      np.nan),
            "SwStr_pct": data.get("SwStr_pct", np.nan),
            "K_pct":     data.get("K_pct",     np.nan),
            "xFIP":      data.get("xFIP",      np.nan),
            "GS":        gs,
            "pa":        data.get("pa", 0),
        })

    df = pd.DataFrame(records)
    # Keep only pitchers who actually started at least 1 game or have ≥ 50 PA faced
    df = df[(df["GS"] >= 1) | (df["pa"] >= 50)].copy()
    df = (df.sort_values("GS", ascending=False)
            .drop_duplicates("name_norm")
            .reset_index(drop=True))
    return df


def fetch_pitcher_stuff(start_year: int = 2015,
                        end_year: int | None = None,
                        cache_path: str = CACHE_PATH) -> pd.DataFrame:
    if end_year is None:
        end_year = pd.Timestamp.today().year

    existing = None
    fetch_years = list(range(start_year, end_year + 1))

    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        if not existing.empty and "year" in existing.columns:
            cached_years = set(existing["year"].unique())
            # Fetch years not yet in cache, plus always re-fetch current year
            fetch_years = [y for y in fetch_years
                           if y not in cached_years or y == end_year]
            if not fetch_years:
                print(f"Pitcher stuff cache up to date (years: {sorted(cached_years)})")
                return existing
            print(f"Cached years: {sorted(cached_years)}; fetching {fetch_years}…")

    frames = []
    for year in fetch_years:
        print(f"  Fetching Statcast pitcher stuff {year}…")
        try:
            df = _fetch_year(year)
            if not df.empty:
                frames.append(df)
                print(f"    → {len(df)} pitchers")
            else:
                print(f"    → no data")
        except Exception as exc:
            print(f"    Warning: {year} failed — {exc}")

    if not frames:
        return existing if existing is not None else pd.DataFrame()

    new_data = pd.concat(frames, ignore_index=True)

    combined = (
        pd.concat([existing, new_data], ignore_index=True)
        if existing is not None else new_data
    )
    # Drop old rows for re-fetched years
    if existing is not None:
        for yr in fetch_years:
            old_idx = existing[existing["year"] == yr].index
            combined = combined.drop(index=old_idx, errors="ignore")
        combined = combined.reset_index(drop=True)

    combined = (combined.sort_values(["year", "name_norm"])
                        .drop_duplicates(["name_norm", "year"])
                        .reset_index(drop=True))
    combined.to_csv(cache_path, index=False)
    print(f"Saved {len(combined):,} rows → {cache_path}")
    return combined


def get_pitcher_stuff(name_norm: str,
                       team: str,
                       year: int,
                       stuff_df: pd.DataFrame,
                       min_pa: int = 150) -> dict:
    """
    Return pitch stuff for a pitcher.

    When current-year PA < min_pa (early season small sample), blend with the
    most recent prior-year row weighted by PA so a 2-start xFIP doesn't dominate.
    Falls back to team median if the pitcher isn't found at all.
    """
    empty = {"FBv": np.nan, "SwStr_pct": np.nan, "K_pct": np.nan,
             "BB_pct": np.nan, "xFIP": np.nan, "Throws": None}

    if stuff_df is None or stuff_df.empty:
        return empty

    pitcher_rows = stuff_df[stuff_df["name_norm"] == name_norm].sort_values("year")

    if not pitcher_rows.empty:
        cur = pitcher_rows[pitcher_rows["year"] == year]
        prior = pitcher_rows[pitcher_rows["year"] < year]

        if cur.empty and not prior.empty:
            # No current-year data — use most recent prior year
            r = prior.iloc[-1]
            return {k: r.get(k, np.nan) for k in empty}

        if not cur.empty:
            r_cur = cur.iloc[0]
            cur_pa = int(r_cur.get("pa", 0) or 0)

            if cur_pa >= min_pa:
                # Enough sample — use current year as-is
                return {k: r_cur.get(k, np.nan) for k in empty}

            if prior.empty:
                # Small sample, no prior seasons — blend with league median to avoid noise.
                # Use pitchers with >= 40 PA as the anchor (works early in the season).
                anchor_rows = stuff_df[(stuff_df["year"] == year) & (stuff_df["pa"] >= 40)
                                       & (stuff_df["name_norm"] != name_norm)]
                if anchor_rows.empty:
                    anchor_rows = stuff_df[stuff_df["year"] == year]
                med_pa = 300  # weight for the median anchor (≈ half a season)
                total = cur_pa + med_pa
                result = {}
                for col in ["FBv", "SwStr_pct", "K_pct", "xFIP"]:
                    c_val = r_cur.get(col, np.nan)
                    m_val = float(anchor_rows[col].median()) if col in anchor_rows and anchor_rows[col].notna().any() else np.nan
                    if pd.notna(c_val) and pd.notna(m_val):
                        result[col] = (c_val * cur_pa + m_val * med_pa) / total
                    elif pd.notna(c_val):
                        result[col] = float(c_val)
                    else:
                        result[col] = m_val if pd.notna(m_val) else np.nan
                result["BB_pct"] = np.nan
                result["Throws"] = r_cur.get("Throws")
                return result

            # Small sample — blend current year with most recent prior year
            r_pri = prior.iloc[-1]
            pri_pa = min(int(r_pri.get("pa", 0) or 0), 600)  # cap prior weight at ~1 season
            total = cur_pa + pri_pa
            result = {}
            for col in ["FBv", "SwStr_pct", "K_pct", "xFIP"]:
                c_val = r_cur.get(col, np.nan)
                p_val = r_pri.get(col, np.nan)
                if pd.notna(c_val) and pd.notna(p_val):
                    result[col] = (c_val * cur_pa + p_val * pri_pa) / total
                elif pd.notna(c_val):
                    result[col] = float(c_val)
                elif pd.notna(p_val):
                    result[col] = float(p_val)
                else:
                    result[col] = np.nan
            result["BB_pct"] = np.nan
            result["Throws"] = r_cur.get("Throws") or r_pri.get("Throws")
            return result

    # Pitcher not found — team median fallback
    team_rows = stuff_df[(stuff_df["team"] == team) & (stuff_df["year"] == year)]
    if team_rows.empty:
        return empty

    result = {}
    for col in ["FBv", "SwStr_pct", "K_pct", "xFIP"]:
        if col in team_rows.columns:
            result[col] = float(team_rows[col].median()) if team_rows[col].notna().any() else np.nan
        else:
            result[col] = np.nan
    result["BB_pct"] = np.nan
    result["Throws"] = None
    return result


def backfill_handedness(cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Look up pitcher handedness (R/L) from the MLB Stats API for any row
    in pitcher_stuff.csv that has Throws == NaN.  Results are cached in
    data/pitcher_handedness.json so API calls only happen once per pitcher.
    """
    import json
    try:
        import statsapi
    except ImportError:
        print("  statsapi not installed — skipping handedness backfill")
        return pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame()

    hand_cache_path = os.path.join(DATA_DIR, "pitcher_handedness.json")
    hand_cache: dict[str, str | None] = {}
    if os.path.exists(hand_cache_path):
        with open(hand_cache_path) as f:
            hand_cache = json.load(f)

    df = pd.read_csv(cache_path)
    missing_names = df.loc[df["Throws"].isna(), "name_norm"].unique().tolist()
    to_lookup = [n for n in missing_names if n not in hand_cache and isinstance(n, str) and n.strip()]

    if to_lookup:
        print(f"  Looking up handedness for {len(to_lookup)} pitchers via MLB Stats API...")
        for name_norm in to_lookup:
            try:
                # Step 1: find player ID by name
                results = statsapi.lookup_player(name_norm)
                found = None
                for p in results:
                    if (p.get("primaryPosition") or {}).get("code") == "1":
                        pid = p.get("id")
                        if pid:
                            # Step 2: fetch full person record for pitchHand
                            person_data = statsapi.get("person", {"personId": pid})
                            person = (person_data.get("people") or [{}])[0]
                            hand = (person.get("pitchHand") or {}).get("code")
                            if hand in ("R", "L"):
                                found = hand
                                break
                hand_cache[name_norm] = found
                time.sleep(0.08)
            except Exception:
                hand_cache[name_norm] = None

        with open(hand_cache_path, "w") as f:
            json.dump(hand_cache, f)
        print(f"  Handedness cache updated: {sum(1 for v in hand_cache.values() if v)} pitchers resolved")

    df["Throws"] = df.apply(
        lambda r: hand_cache.get(r["name_norm"], r["Throws"])
        if pd.isna(r["Throws"]) else r["Throws"],
        axis=1,
    )
    df.to_csv(cache_path, index=False)
    resolved = df["Throws"].notna().sum()
    print(f"  Throws populated for {resolved:,}/{len(df):,} pitcher-seasons")
    return df


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    df = fetch_pitcher_stuff(start_year=start)
    print(f"\nDone. {len(df):,} pitcher-seasons in {CACHE_PATH}")
    if not df.empty:
        show_cols = [c for c in ["name_norm", "team", "year", "FBv", "SwStr_pct", "K_pct", "xFIP"] if c in df.columns]
        print(df[show_cols].head(10).to_string())
    print("\nBackfilling pitcher handedness...")
    backfill_handedness()
