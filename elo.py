"""
NBA-God — Elo Rating System
=============================
Calibrated for basketball's 82-game season:
  - K=12 (moderate — more games than football, fewer than baseball)
  - Home advantage = 40 Elo points (~56% home win rate in NBA)
  - Season reversion = 33% toward mean (teams regress between seasons)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Elo parameters (basketball-calibrated) ────────────────────────────────────
ELO_INITIAL = 1500
ELO_K = 12
ELO_HOME_ADVANTAGE = 40
ELO_SEASON_REVERT = 0.33


class EloSystem:
    """Elo rating system calibrated for NBA."""

    def __init__(self, k: float = ELO_K, home_adv: float = ELO_HOME_ADVANTAGE,
                 revert: float = ELO_SEASON_REVERT):
        self.k = k
        self.home_adv = home_adv
        self.revert = revert
        self.ratings: dict[int, float] = {}
        self.history: list[dict] = []

    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, ELO_INITIAL)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """P(A wins) given Elo ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, home_id: int, away_id: int, home_won: bool,
               margin: int = 0) -> tuple[float, float]:
        """
        Update ratings after a game.
        Returns (home_new_rating, away_new_rating).
        """
        home_elo = self.get_rating(home_id)
        away_elo = self.get_rating(away_id)

        # Home advantage
        home_adj = home_elo + self.home_adv
        expected_home = self.expected_score(home_adj, away_elo)

        # Margin-of-victory multiplier (log-based, capped)
        abs_margin = abs(margin) if margin else 0
        mov_mult = np.log(max(abs_margin, 1) + 1) * 0.7 + 0.7
        mov_mult = min(mov_mult, 2.5)

        actual_home = 1.0 if home_won else 0.0
        delta = self.k * mov_mult * (actual_home - expected_home)

        new_home = home_elo + delta
        new_away = away_elo - delta

        self.ratings[home_id] = new_home
        self.ratings[away_id] = new_away

        return new_home, new_away

    def season_reset(self):
        """Revert all ratings toward mean between seasons."""
        for team_id in self.ratings:
            self.ratings[team_id] = (
                ELO_INITIAL * self.revert +
                self.ratings[team_id] * (1 - self.revert)
            )


def build_elo_ratings(games_df: pd.DataFrame) -> tuple:
    """
    Build Elo ratings from a games DataFrame.
    Returns (EloSystem, games_df_with_elo_columns).
    """
    print("[Elo] Building ratings...")
    elo = EloSystem()

    games = games_df.sort_values("game_date").copy()
    home_elos, away_elos, elo_diffs = [], [], []

    prev_season = None
    for _, game in games.iterrows():
        season = game["season"]
        if prev_season is not None and season != prev_season:
            elo.season_reset()
        prev_season = season

        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        # Record pre-game Elo
        h_elo = elo.get_rating(home_id)
        a_elo = elo.get_rating(away_id)
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        elo_diffs.append(h_elo - a_elo)

        # Update
        home_won = game["home_win"] == 1
        margin = game.get("margin", 0) or 0
        elo.update(home_id, away_id, home_won, margin)

    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_diff"] = elo_diffs

    print(f"  Processed {len(games):,} games")
    print(f"  Teams rated: {len(elo.ratings)}")

    # Top/bottom teams
    sorted_ratings = sorted(elo.ratings.items(), key=lambda x: -x[1])
    print(f"  Top 5 Elo: {[(tid, round(r, 1)) for tid, r in sorted_ratings[:5]]}")

    return elo, games


if __name__ == "__main__":
    import sqlite3
    ROOT = Path(__file__).parent
    db_path = ROOT / "phase1_data" / "output" / "nba_god.db"
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        games = pd.read_sql("SELECT * FROM games ORDER BY game_date", conn)
        conn.close()
        elo, games_with_elo = build_elo_ratings(games)
        print(f"\nFinal ratings for {len(elo.ratings)} teams computed.")
    else:
        print("No database found. Run phase1_data/ingest.py first.")
