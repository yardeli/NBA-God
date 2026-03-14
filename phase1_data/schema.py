"""
Phase 1: Unified Data Schema — Single canonical schema for all NBA game data.

Every game from every source gets normalized into this schema.
Sources may provide different levels of detail, tracked by `data_completeness_tier`:
  - Tier 1: Full box score (shooting splits, rebounds, assists, steals, blocks, TOs)
  - Tier 2: Basic stats (points, FG%, rebounds, assists)
  - Tier 3: Score only (date, teams, final score — that's it)

Era metadata captures rule changes that affect stat interpretation:
  - early: 1946-1954 (BAA/early NBA, no shot clock)
  - shot_clock: 1955-1978 (24-second clock introduced, ABA merger 1976)
  - 3pt_intro: 1979-1994 (3-point line introduced, physical play era)
  - modern: 1995-2003 (zone defense legalized, hand-check era)
  - analytics: 2004-2015 (hand-check rule removed, pace increases)
  - pace_space: 2016+ (3-point revolution, pace-and-space offense)
"""
import sqlite3
from pathlib import Path

SCHEMA_VERSION = "1.0"

# ── Game types ──
GAME_TYPES = {
    "regular": "Regular Season",
    "play_in": "Play-In Tournament",
    "first_round": "First Round (Best-of-7)",
    "conference_semis": "Conference Semifinals",
    "conference_finals": "Conference Finals",
    "nba_finals": "NBA Finals",
    "all_star": "All-Star Game",
}

# ── Era definitions ──
ERA_DEFINITIONS = {
    "early":       (1946, 1954),
    "shot_clock":  (1955, 1978),
    "3pt_intro":   (1979, 1994),
    "modern":      (1995, 2003),
    "analytics":   (2004, 2015),
    "pace_space":  (2016, 9999),
}

def get_era(season: int) -> str:
    for name, (start, end) in ERA_DEFINITIONS.items():
        if start <= season <= end:
            return name
    return "pace_space"

def get_era_flags(season: int) -> dict:
    return {
        "era": get_era(season),
        "has_3pt_line": season >= 1980,
        "shot_clock": season >= 1955,
        "hand_check_allowed": season < 2004,
        "zone_defense_legal": season >= 2002,
        "play_in_tournament": season >= 2021,
        "shortened_3pt_line": 1995 <= season <= 1997,  # NBA shortened 3pt 94-95 to 96-97
        "games_per_season": (
            82 if season >= 1968 else
            (80 if season >= 1962 else
             (79 if season >= 1960 else 72))
        ) if season != 2020 and season != 2021 else (
            72 if season == 2021 else 0  # Bubble 2020 varied, 2021 shortened
        ),
    }


# ── SQLite Schema ──
CREATE_TABLES_SQL = """
-- Teams master table
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    abbreviation TEXT,
    conference TEXT,  -- Eastern or Western
    division TEXT,
    city TEXT,
    espn_id INTEGER,
    bbref_id TEXT,
    is_active INTEGER DEFAULT 1,
    first_season INTEGER,
    last_season INTEGER
);

-- Team name aliases for deduplication
CREATE TABLE IF NOT EXISTS team_aliases (
    alias TEXT PRIMARY KEY,
    team_id INTEGER NOT NULL,
    source TEXT,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Franchise history (teams move/rename)
CREATE TABLE IF NOT EXISTS franchise_history (
    team_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    name TEXT NOT NULL,
    city TEXT,
    conference TEXT,
    division TEXT,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Unified games table
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    game_type TEXT NOT NULL DEFAULT 'regular',

    -- Teams
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team_name TEXT NOT NULL,
    away_team_name TEXT NOT NULL,

    -- Score
    home_pts INTEGER NOT NULL,
    away_pts INTEGER NOT NULL,
    home_win INTEGER NOT NULL,
    overtime_periods INTEGER DEFAULT 0,

    -- Postseason context
    series_game_num INTEGER,
    series_status TEXT,

    -- Basic box score
    home_fgm INTEGER, home_fga INTEGER,
    away_fgm INTEGER, away_fga INTEGER,
    home_fg3m INTEGER, home_fg3a INTEGER,
    away_fg3m INTEGER, away_fg3a INTEGER,
    home_ftm INTEGER, home_fta INTEGER,
    away_ftm INTEGER, away_fta INTEGER,
    home_oreb INTEGER, home_dreb INTEGER, home_reb INTEGER,
    away_oreb INTEGER, away_dreb INTEGER, away_reb INTEGER,
    home_ast INTEGER, home_stl INTEGER, home_blk INTEGER, home_tov INTEGER,
    away_ast INTEGER, away_stl INTEGER, away_blk INTEGER, away_tov INTEGER,
    home_pf INTEGER, away_pf INTEGER,

    -- Advanced (Tier 1)
    home_pace REAL,
    away_pace REAL,
    home_off_eff REAL,
    away_off_eff REAL,
    home_def_eff REAL,
    away_def_eff REAL,

    -- Context
    arena_name TEXT,
    attendance INTEGER,
    game_duration_minutes INTEGER,

    -- Computed
    margin INTEGER,
    total_pts INTEGER,

    -- Metadata
    era TEXT NOT NULL,
    data_completeness_tier INTEGER NOT NULL DEFAULT 3,
    data_source TEXT NOT NULL,
    source_game_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_games_type ON games(game_type);

-- Arenas
CREATE TABLE IF NOT EXISTS arenas (
    arena_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team_id INTEGER,
    capacity INTEGER,
    city TEXT,
    FOREIGN KEY (arena_id) REFERENCES teams(team_id)
);

-- Team season stats (pre-computed aggregates)
CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    games_played INTEGER,
    wins INTEGER,
    losses INTEGER,
    win_pct REAL,
    pts_per_game REAL,
    opp_pts_per_game REAL,
    point_differential REAL,
    pace REAL,
    off_eff REAL,
    def_eff REAL,
    net_eff REAL,
    efg_pct REAL,
    to_rate REAL,
    orb_rate REAL,
    ft_rate REAL,
    pythagorean_wins REAL,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Data source tracking
CREATE TABLE IF NOT EXISTS data_sources (
    source_name TEXT PRIMARY KEY,
    games_loaded INTEGER DEFAULT 0,
    seasons_covered TEXT,
    last_updated TEXT,
    notes TEXT
);

-- Experiment log
CREATE TABLE IF NOT EXISTS experiment_log (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    phase TEXT,
    description TEXT,
    parameters TEXT,
    metrics TEXT,
    notes TEXT
);
"""


def init_database(db_path: str = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = str(Path(__file__).parent / "output" / "nba_god.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
    print(f"[Schema] Database initialized: {db_path}")
    return conn


def compute_completeness_tier(game: dict) -> int:
    box_fields = ["home_fgm", "home_fga", "home_ast", "home_reb", "home_tov", "home_stl"]
    has_box = sum(1 for f in box_fields if game.get(f) is not None) >= 5
    if has_box:
        return 1
    basic_fields = ["home_fgm", "home_fga", "home_reb"]
    has_basic = sum(1 for f in basic_fields if game.get(f) is not None) >= 2
    if has_basic:
        return 2
    return 3


if __name__ == "__main__":
    conn = init_database()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables created: {tables}")
    conn.close()
