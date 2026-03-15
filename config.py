"""
NBA-God v1.0 — Configuration
Professional NBA prediction system covering 1946-2025.
Predicts ANY NBA game: regular season, play-in, playoffs, NBA Finals.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIZ_DIR = BASE_DIR / "visualizations"

# ─── ESPN API ────────────────────────────────────────────────────────────────
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_TEAMS = f"{ESPN_BASE}/teams"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_STANDINGS = f"{ESPN_BASE}/standings"
ESPN_RANKINGS = f"{ESPN_BASE}/rankings"
ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# ─── News/Injury RSS Feeds ──────────────────────────────────────────────────
RSS_FEEDS = {
    "espn_nba": "https://www.espn.com/espn/rss/nba/news",
    "cbs_nba": "https://www.cbssports.com/rss/headlines/nba/",
    "rotowire_nba": "https://www.rotowire.com/rss/basketball-news.xml",
}
ROTOWIRE_INJURY_URL = "https://www.rotowire.com/basketball/injury-report.php"

# ─── Seasons ─────────────────────────────────────────────────────────────────
HISTORICAL_SEASONS = list(range(1980, 2027))
CURRENT_SEASON = 2026
MIN_SEASON = 1980

# ─── Conferences & Divisions (2024 alignment) ───────────────────────────────
EAST_ATLANTIC = {"Celtics", "Nets", "Knicks", "76ers", "Raptors"}
EAST_CENTRAL = {"Bulls", "Cavaliers", "Pistons", "Pacers", "Bucks"}
EAST_SOUTHEAST = {"Hawks", "Hornets", "Heat", "Magic", "Wizards"}
WEST_NORTHWEST = {"Nuggets", "Timberwolves", "Thunder", "Trail Blazers", "Jazz"}
WEST_PACIFIC = {"Warriors", "Clippers", "Lakers", "Suns", "Kings"}
WEST_SOUTHWEST = {"Mavericks", "Rockets", "Grizzlies", "Pelicans", "Spurs"}

# ─── Features ────────────────────────────────────────────────────────────────
TEAM_ROLLING_FEATURES = [
    "pts_pg", "opp_pts_pg", "margin_pg", "win_pct", "momentum",
]

MATCHUP_FEATURES = [
    "elo_diff", "point_diff", "win_pct_diff",
    "pts_pg_diff", "opp_pts_pg_diff",
    "home_away", "rest_days_diff",
    "off_eff_diff", "def_eff_diff", "net_eff_diff",
    "pace_diff", "efg_pct_diff", "to_rate_diff",
    "orb_rate_diff", "ft_rate_diff", "3pt_rate_diff",
    "ast_to_diff", "back_to_back_diff",
    "momentum_diff",
]

# ─── Elo System ──────────────────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K = 12          # Moderate K for 82-game season
ELO_HOME_ADVANTAGE = 40  # ~56% home win rate in NBA
ELO_SEASON_REVERT = 0.33

# ─── Model ───────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 5,
    "n_estimators": 600,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

LOGISTIC_PARAMS = {
    "C": 0.5,
    "max_iter": 2000,
    "random_state": 42,
}

ENSEMBLE_WEIGHT_XGB = 0.80  # Phase 2 optimization: 80% XGBoost, 20% LR (Brier: 0.16374)
ENSEMBLE_WEIGHT_LR = 0.20   # Better probability calibration at extremes

# ─── Rolling Window ─────────────────────────────────────────────────────────
ROLLING_WINDOW = 10   # Games for rolling averages (82-game season)
MOMENTUM_WINDOW = 5   # Recent games for momentum signal

# ─── Spread Sigma ────────────────────────────────────────────────────────────
NBA_SIGMA = 10.5  # Points: how many points correspond to one unit of log-odds

# ─── Request Settings ────────────────────────────────────────────────────────
REQUEST_DELAY = 1.0
REQUEST_TIMEOUT = 15
USER_AGENT = "NBA-God/1.0 (research project)"
