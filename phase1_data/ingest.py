"""
NBA-God — Phase 1: Data Acquisition & Schema Design
=====================================================
Ingests NBA game data from multiple sources into a unified SQLite database.

Data sources (in priority order):
  1. ESPN NBA API (scoreboard + historical, richest data, free)
  2. Basketball Reference CSV game logs (supplemental/historical)

Coverage: 1946-2025 (80 years, 80K+ games)
  - Tier 1: Full box score (shooting, rebounds, assists, etc.) — 2000+
  - Tier 2: Basic stats (points, FG%, rebounds) — 1980-1999
  - Tier 3: Score only (points, date, teams) — 1946-1979
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "phase1_data" / "output" / "nba_god.db"
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "phase1_data" / "output"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from phase1_data.schema import (
    init_database, get_era, get_era_flags, compute_completeness_tier, ERA_DEFINITIONS
)
from phase1_data.team_normalization import TeamNormalizer

# ── ESPN NBA config ──────────────────────────────────────────────────────────
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
REQUEST_DELAY = 0.5
USER_AGENT = "NBA-God/1.0 (research project)"


# ── ESPN data fetching ───────────────────────────────────────────────────────

def fetch_espn_scoreboard(date_str: str) -> list:
    """
    Fetch NBA scoreboard for a given date from ESPN API.
    date_str: YYYYMMDD format
    """
    url = f"{ESPN_BASE}/scoreboard"
    params = {"dates": date_str}
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        games = []
        for event in data.get("events", []):
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status == "STATUS_FINAL":
                games.append(event)
        return games
    except Exception as e:
        print(f"    [WARN] ESPN error for {date_str}: {e}")
        return []


def parse_espn_game(event: dict, normalizer: TeamNormalizer,
                    game_type_str: str = "regular") -> dict:
    """Parse a single ESPN event into our schema format."""
    comp = event.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return {}

    home_data = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away_data = next((c for c in competitors if c.get("homeAway") == "away"), {})

    home_name = home_data.get("team", {}).get("displayName", "Unknown")
    away_name = away_data.get("team", {}).get("displayName", "Unknown")

    home_canonical, home_id = normalizer.resolve(home_name, source="espn")
    away_canonical, away_id = normalizer.resolve(away_name, source="espn")

    home_pts = int(home_data.get("score", 0) or 0)
    away_pts = int(away_data.get("score", 0) or 0)

    game_date = event.get("date", "")[:10]  # YYYY-MM-DD
    season = int(game_date[:4]) if game_date else 0
    # NBA season spans two years: games after June are prior season
    if game_date:
        month = int(game_date[5:7])
        if month >= 10:
            season = int(game_date[:4]) + 1  # e.g., Oct 2024 = 2025 season
        elif month <= 6:
            season = int(game_date[:4])  # e.g., Mar 2025 = 2025 season

    # Extract stats from competitor statistics
    home_stats = _extract_team_stats(home_data)
    away_stats = _extract_team_stats(away_data)

    venue = comp.get("venue", {}).get("fullName", "")

    # Detect overtime
    ot_periods = 0
    status_detail = comp.get("status", {}).get("type", {}).get("detail", "")
    if "OT" in status_detail:
        try:
            if status_detail.startswith("Final/"):
                ot_str = status_detail.split("/")[1].strip()
                if ot_str == "OT":
                    ot_periods = 1
                else:
                    ot_periods = int(ot_str.replace("OT", ""))
        except Exception:
            ot_periods = 1

    era_flags = get_era_flags(season)
    event_id = event.get("id", "0")

    # Detect playoff game type from season type
    season_type = event.get("season", {}).get("type", 2)  # 2=regular, 3=postseason
    if season_type == 3:
        series_text = event.get("series", {}).get("summary", "").lower()
        if "final" in series_text:
            game_type_str = "nba_finals"
        elif "conference final" in series_text:
            game_type_str = "conference_finals"
        elif "conference semi" in series_text:
            game_type_str = "conference_semis"
        else:
            game_type_str = "first_round"

    row = {
        "game_id": f"nba_{event_id}",
        "season": season,
        "game_date": game_date,
        "game_type": game_type_str,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team_name": home_canonical,
        "away_team_name": away_canonical,
        "home_pts": home_pts,
        "away_pts": away_pts,
        "home_win": 1 if home_pts > away_pts else 0,
        "overtime_periods": ot_periods,
        "arena_name": venue,
        "margin": home_pts - away_pts,
        "total_pts": home_pts + away_pts,
        "era": era_flags["era"],
        "data_completeness_tier": 2,
        "data_source": "espn",
        "source_game_id": str(event_id),
    }

    # Add box score stats if available
    for prefix, stats in [("home", home_stats), ("away", away_stats)]:
        for key, val in stats.items():
            row[f"{prefix}_{key}"] = val

    row["data_completeness_tier"] = compute_completeness_tier(row)
    return row


def _extract_team_stats(competitor: dict) -> dict:
    """Extract box score stats from ESPN competitor data."""
    stats = {}
    stat_list = competitor.get("statistics", [])
    if not stat_list:
        return stats

    # ESPN statistics are in a flat list with names
    stat_map = {}
    for s in stat_list:
        stat_map[s.get("name", "")] = s.get("displayValue", "")

    def safe_int(key):
        v = stat_map.get(key, "")
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    def safe_split(key, idx):
        v = stat_map.get(key, "")
        try:
            parts = v.split("-")
            return int(parts[idx])
        except (ValueError, TypeError, IndexError):
            return None

    stats["fgm"] = safe_split("fieldGoalsMade-fieldGoalsAttempted", 0) or safe_int("fieldGoalsMade")
    stats["fga"] = safe_split("fieldGoalsMade-fieldGoalsAttempted", 1) or safe_int("fieldGoalsAttempted")
    stats["fg3m"] = safe_split("threePointFieldGoalsMade-threePointFieldGoalsAttempted", 0) or safe_int("threePointFieldGoalsMade")
    stats["fg3a"] = safe_split("threePointFieldGoalsMade-threePointFieldGoalsAttempted", 1) or safe_int("threePointFieldGoalsAttempted")
    stats["ftm"] = safe_split("freeThrowsMade-freeThrowsAttempted", 0) or safe_int("freeThrowsMade")
    stats["fta"] = safe_split("freeThrowsMade-freeThrowsAttempted", 1) or safe_int("freeThrowsAttempted")
    stats["oreb"] = safe_int("offensiveRebounds")
    stats["dreb"] = safe_int("defensiveRebounds")
    stats["reb"] = safe_int("totalRebounds") or safe_int("rebounds")
    stats["ast"] = safe_int("assists")
    stats["stl"] = safe_int("steals")
    stats["blk"] = safe_int("blocks")
    stats["tov"] = safe_int("turnovers") or safe_int("totalTurnovers")
    stats["pf"] = safe_int("fouls") or safe_int("personalFouls")

    return {k: v for k, v in stats.items() if v is not None}


# ── ESPN season ingestion ────────────────────────────────────────────────────

def ingest_espn_season(normalizer: TeamNormalizer, season: int) -> list:
    """Ingest all games for a single NBA season from ESPN."""
    # NBA season runs Oct-June
    start_year = season - 1
    all_games = []

    # Regular season: October through mid-April
    # Playoffs: April through June
    start_date = datetime(start_year, 10, 1)
    end_date = datetime(season, 6, 30)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        events = fetch_espn_scoreboard(date_str)
        for event in events:
            row = parse_espn_game(event, normalizer)
            if row and row.get("season", 0) > 0:
                all_games.append(row)
        current += timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    return all_games


def ingest_espn(conn: sqlite3.Connection, normalizer: TeamNormalizer,
                start_season: int = 2010, end_season: int = 2025):
    """Ingest games from ESPN for the given season range."""
    print(f"\nIngesting from ESPN NBA API ({start_season}-{end_season})...")

    all_games = []
    for season in range(start_season, end_season + 1):
        print(f"  Season {season}...", end=" ", flush=True)
        season_games = ingest_espn_season(normalizer, season)
        all_games.extend(season_games)
        print(f"{len(season_games)} games")

    if all_games:
        df = pd.DataFrame(all_games)
        print(f"  Total ESPN games: {len(df):,}")
        return df
    return pd.DataFrame()


# ── CSV data loading (Basketball Reference / historical dumps) ───────────────

def load_csv_gamelogs(normalizer: TeamNormalizer) -> pd.DataFrame:
    """
    Load game data from CSV files in data/raw/csv/.
    Supports common formats from Basketball Reference.
    """
    csv_dir = RAW_DIR / "csv"
    if not csv_dir.exists():
        print("  [INFO] No CSV data found. Skipping.")
        return pd.DataFrame()

    print("  Loading CSV game logs...")
    all_rows = []

    for csv_file in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"    [WARN] Could not read {csv_file.name}: {e}")
            continue

        col_map = _detect_csv_columns(df)
        if not col_map:
            print(f"    [WARN] Could not detect columns in {csv_file.name}")
            continue

        for _, r in df.iterrows():
            try:
                home_name = str(r[col_map["home_team"]])
                away_name = str(r[col_map["away_team"]])
                home_canonical, home_id = normalizer.resolve(home_name, "csv")
                away_canonical, away_id = normalizer.resolve(away_name, "csv")

                game_date = str(r[col_map["date"]])
                season = int(game_date[:4])
                month = int(game_date[5:7]) if len(game_date) > 6 else 1
                if month >= 10:
                    season += 1

                home_pts = int(r[col_map["home_score"]])
                away_pts = int(r[col_map["away_score"]])

                era_flags = get_era_flags(season)

                row = {
                    "game_id": f"csv_{game_date}_{home_name}_{away_name}",
                    "season": season,
                    "game_date": game_date,
                    "game_type": "regular",
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_team_name": home_canonical,
                    "away_team_name": away_canonical,
                    "home_pts": home_pts,
                    "away_pts": away_pts,
                    "home_win": 1 if home_pts > away_pts else 0,
                    "overtime_periods": 0,
                    "margin": home_pts - away_pts,
                    "total_pts": home_pts + away_pts,
                    "era": era_flags["era"],
                    "data_completeness_tier": 3,
                    "data_source": "csv",
                    "source_game_id": f"{game_date}_{home_name}",
                }
                row["data_completeness_tier"] = compute_completeness_tier(row)
                all_rows.append(row)
            except Exception:
                continue

    if all_rows:
        print(f"    {len(all_rows):,} CSV games loaded.")
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def _detect_csv_columns(df: pd.DataFrame) -> dict:
    """Auto-detect column names for common CSV formats."""
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for key in ["date", "game_date", "gamedate", "dt"]:
        if key in cols:
            mapping["date"] = cols[key]
            break
    for key in ["home_team", "home", "hometeam", "hm_team", "home_team_name"]:
        if key in cols:
            mapping["home_team"] = cols[key]
            break
    for key in ["away_team", "away", "awayteam", "vis_team", "away_team_name", "visitor"]:
        if key in cols:
            mapping["away_team"] = cols[key]
            break
    for key in ["home_score", "home_pts", "hm_pts", "pts_home", "home_points"]:
        if key in cols:
            mapping["home_score"] = cols[key]
            break
    for key in ["away_score", "away_pts", "vis_pts", "pts_away", "away_points", "visitor_pts"]:
        if key in cols:
            mapping["away_score"] = cols[key]
            break

    if len(mapping) >= 5:
        return mapping
    return {}


# ── Merge and deduplicate ────────────────────────────────────────────────────

def merge_and_deduplicate(*dataframes) -> pd.DataFrame:
    """
    Merge games from multiple sources, dedup by (date, home, away).
    Priority: espn > csv (higher quality data preferred).
    """
    print("\n  Merging and deduplicating across sources...")
    dfs = [df for df in dataframes if len(df) > 0]
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    source_priority = {"espn": 0, "bbref": 1, "csv": 2}
    combined["_priority"] = combined["data_source"].map(source_priority).fillna(9)

    combined["_dedup_key"] = (
        combined["game_date"].astype(str) + "_" +
        combined["home_team_name"].astype(str) + "_" +
        combined["away_team_name"].astype(str)
    )

    combined = (combined.sort_values("_priority")
                .drop_duplicates(subset="_dedup_key", keep="first")
                .drop(columns=["_priority", "_dedup_key"])
                .reset_index(drop=True))

    print(f"  After dedup: {len(combined):,} games")
    return combined


# ── Write to database ────────────────────────────────────────────────────────

def write_games_to_db(conn: sqlite3.Connection, games_df: pd.DataFrame):
    """Write games DataFrame to SQLite database."""
    print(f"\n  Writing {len(games_df):,} games to database...")

    cursor = conn.execute("PRAGMA table_info(games)")
    db_columns = {row[1] for row in cursor.fetchall()}
    write_cols = [c for c in games_df.columns if c in db_columns]

    games_df[write_cols].to_sql("games", conn, if_exists="replace", index=False)
    print(f"  Done. {len(games_df):,} games written.")


# ── Register teams ──────────────────────────────────────────────────────────

def register_teams(conn: sqlite3.Connection, normalizer: TeamNormalizer,
                   games_df: pd.DataFrame):
    """Register all teams and aliases into the database."""
    print("\n  Registering teams...")

    teams_data = []
    for canonical, tid in normalizer.canonical_to_id.items():
        teams_data.append({
            "team_id": tid,
            "canonical_name": canonical,
            "is_active": 1,
        })

    teams_df = pd.DataFrame(teams_data)
    teams_df.to_sql("teams", conn, if_exists="replace", index=False)

    aliases_data = []
    for alias, canonical in normalizer.alias_to_canonical.items():
        tid = normalizer.canonical_to_id.get(canonical, 0)
        aliases_data.append({
            "alias": alias,
            "team_id": tid,
            "source": "normalizer",
        })
    if aliases_data:
        pd.DataFrame(aliases_data).to_sql(
            "team_aliases", conn, if_exists="replace", index=False
        )

    print(f"  {len(teams_data)} teams, {len(aliases_data)} aliases registered.")


# ── Summary report ───────────────────────────────────────────────────────────

def generate_report(games_df: pd.DataFrame) -> dict:
    """Generate summary of ingested data."""
    print("\nGenerating summary report...")

    report = {
        "total_games": len(games_df),
        "season_range": {
            "earliest": int(games_df["season"].min()),
            "latest": int(games_df["season"].max()),
            "total_seasons": int(games_df["season"].nunique()),
        },
        "games_by_type": games_df["game_type"].value_counts().to_dict(),
        "games_by_era": games_df["era"].value_counts().to_dict(),
        "games_by_source": games_df["data_source"].value_counts().to_dict(),
        "data_tier_counts": games_df["data_completeness_tier"].value_counts().to_dict(),
        "games_by_decade": games_df.assign(
            decade=(games_df["season"] // 10 * 10).astype(str) + "s"
        )["decade"].value_counts().sort_index().to_dict(),
    }

    report_path = REPORT_DIR / "phase1_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  PHASE 1 INGESTION SUMMARY")
    print("=" * 60)
    print(f"\n  Seasons: {report['season_range']['earliest']}-"
          f"{report['season_range']['latest']} "
          f"({report['season_range']['total_seasons']} seasons)")

    print("\n  Games by type:")
    for gtype, cnt in sorted(report["games_by_type"].items(), key=lambda x: -x[1]):
        print(f"    {gtype:<20} {cnt:>7,}")
    print(f"    {'TOTAL':<20} {report['total_games']:>7,}")

    print("\n  Games by era:")
    for era, cnt in sorted(report["games_by_era"].items()):
        print(f"    {era:<20} {cnt:>7,}")

    print("\n  Data completeness:")
    tier_labels = {
        1: "Full box score (shooting, rebounds, assists, etc.)",
        2: "Basic stats (points, FG%, rebounds)",
        3: "Score only",
    }
    for tier, cnt in sorted(report["data_tier_counts"].items()):
        pct = cnt / report["total_games"] * 100
        label = tier_labels.get(int(tier), "Unknown")
        print(f"    Tier {tier}: {cnt:>7,} ({pct:.1f}%) -- {label}")

    print(f"\n  Games by source:")
    for src, cnt in sorted(report["games_by_source"].items(), key=lambda x: -x[1]):
        print(f"    {src:<20} {cnt:>7,}")

    print(f"\n  Full report: {report_path}")
    print("=" * 60)
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\nNBA-God — Phase 1: Data Ingestion")
    print(f"DB path: {DB_PATH}")
    print(f"Raw data: {RAW_DIR}\n")

    normalizer = TeamNormalizer()
    conn = init_database(str(DB_PATH))

    try:
        # 1. Ingest from ESPN (2010-2025)
        espn_df = ingest_espn(conn, normalizer, start_season=2010, end_season=2025)

        # 2. Load any CSV data (if available)
        csv_df = load_csv_gamelogs(normalizer)

        # 3. Merge and deduplicate
        games_df = merge_and_deduplicate(espn_df, csv_df)

        if len(games_df) == 0:
            print("\n[ERROR] No games loaded from any source!")
            print("  Place data files in:")
            print(f"    - CSV:  {RAW_DIR / 'csv'}/*.csv")
            print("  Or ensure internet access for ESPN API.")
            sys.exit(1)

        # 4. Register teams
        register_teams(conn, normalizer, games_df)

        # 5. Write games
        write_games_to_db(conn, games_df)

        # 6. Save team normalizer state
        normalizer.save(str(REPORT_DIR / "team_names.json"))

        # 7. Log unresolved teams
        print(f"\n{normalizer.get_unresolved_report()}")

        # 8. Generate report
        report = generate_report(games_df)

        # 9. Record data source
        conn.execute("""
            INSERT OR REPLACE INTO data_sources (source_name, games_loaded, last_updated, notes)
            VALUES (?, ?, ?, ?)
        """, ("phase1_ingest", len(games_df),
              datetime.now().isoformat(), json.dumps(report["games_by_source"])))

        conn.commit()
        print("\nPhase 1 complete. Database ready.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
