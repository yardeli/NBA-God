"""
NBA-God — Daily Predictor
===========================
Generates predictions for today's NBA games with:
  - Win probability (ensemble: 65% XGBoost + 35% Logistic Regression)
  - Point spread via log-odds x NBA_SIGMA
  - Over/under total estimation
  - Edge calculation vs market odds
"""

import json
import pickle
import os
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ROOT = Path(__file__).parent
NBA_SIGMA = 10.5  # Points per unit of log-odds

sys.path.insert(0, str(ROOT))


class DailyPredictor:
    """Lazy-loaded daily prediction engine."""

    def __init__(self):
        self._predictor = None
        self._features = None
        self._team_names = None
        self._elo_system = None

    def _load(self):
        if self._predictor is not None:
            return

        print("[DailyPredictor] Loading model...")
        predictor_path = ROOT / "phase5_deploy" / "output" / "predictor.pkl"
        if predictor_path.exists():
            with open(predictor_path, "rb") as f:
                self._predictor = pickle.load(f)
            self._features = self._predictor.features_df
            self._team_names = self._predictor.team_names
        else:
            # Fallback: load model directly
            model_path = ROOT / "phase3_models" / "output" / "production_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError("No trained model. Run pipeline first.")
            with open(model_path, "rb") as f:
                pkg = pickle.load(f)
            from phase5_deploy.robustness import NBAGodPredictor, load_artifacts
            _, features, team_names = load_artifacts()
            self._predictor = NBAGodPredictor(pkg, team_names, features)
            self._features = features
            self._team_names = team_names

        print(f"  Loaded. {len(self._team_names)} teams.")

    # ESPN sometimes uses abbreviated names
    ESPN_ALIASES = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
    }

    def resolve_team(self, name: str) -> tuple:
        """Resolve team name to (canonical_name, team_id)."""
        self._load()
        name_lower = name.lower().strip()
        # Check ESPN aliases first
        name_lower = self.ESPN_ALIASES.get(name_lower, name_lower)

        # Exact match
        for tid, tname in self._team_names.items():
            if tname.lower() == name_lower:
                return tname, tid

        # Partial match
        matches = []
        for tid, tname in self._team_names.items():
            if name_lower in tname.lower() or tname.lower() in name_lower:
                matches.append((tname, tid))

        if len(matches) == 1:
            return matches[0]

        # City match
        for tid, tname in self._team_names.items():
            parts = tname.lower().split()
            if name_lower in parts or any(name_lower in p for p in parts):
                matches.append((tname, tid))

        if matches:
            return matches[0]

        return name, 0

    def prob_to_spread(self, prob: float) -> float:
        """Convert win probability to point spread."""
        if prob <= 0.01 or prob >= 0.99:
            return 0.0
        # Dampen overconfident probabilities toward 50%
        # Maps model's [0.5-1.0] range to roughly [0.5-0.75] for realistic spreads
        dampened = 0.5 + (prob - 0.5) * 0.45
        log_odds = math.log(dampened / (1 - dampened))
        return round(log_odds * NBA_SIGMA, 1)

    def estimate_total(self, team_a_ppg: float, team_b_ppg: float,
                       team_a_opp_ppg: float, team_b_opp_ppg: float) -> float:
        """Estimate game total points."""
        avg_a = (team_a_ppg + team_b_opp_ppg) / 2
        avg_b = (team_b_ppg + team_a_opp_ppg) / 2
        return round(avg_a + avg_b, 1)

    def predict_game(self, home_team: str, away_team: str,
                     season: int = 2025) -> dict:
        """Predict a single game."""
        self._load()
        # Resolve names first so ESPN abbreviations ("LA Clippers") map correctly
        resolved_home, _ = self.resolve_team(home_team)
        resolved_away, _ = self.resolve_team(away_team)
        try:
            result = self._predictor.predict(resolved_home, resolved_away, season)
        except Exception as e:
            return {
                "home_team": home_team, "away_team": away_team,
                "error": str(e), "home_win_prob": 0.5,
                "away_win_prob": 0.5, "predicted_winner": home_team,
                "confidence": "low", "model_spread": 0.0,
                "model_point_spread": "PICK",
            }

        prob = result["prob_a_wins"]
        spread = self.prob_to_spread(prob)

        return {
            "home_team": result["team_a"],
            "away_team": result["team_b"],
            "home_win_prob": prob,
            "away_win_prob": result["prob_b_wins"],
            "predicted_winner": result["favored"],
            "confidence": result["confidence"],
            "model_spread": spread,
            "model_point_spread": f"{result['team_a']} {spread:+.1f}" if spread != 0 else "PICK",
        }

    def predict_today(self) -> list:
        """Predict all of today's games using ESPN schedule."""
        self._load()

        print(f"\n{'=' * 60}")
        print(f"  NBA-GOD DAILY PREDICTIONS -- {datetime.now().strftime('%B %d, %Y')}")
        print(f"{'=' * 60}")

        games = self._fetch_today_schedule()
        if not games:
            print("  No games scheduled today.")
            return []

        predictions = []
        for game in games:
            pred = self.predict_game(game["home_team"], game["away_team"])
            pred["game_id"] = game.get("game_id", "")
            pred["game_time"] = game.get("game_time", "")
            predictions.append(pred)

        # Sort by confidence
        predictions.sort(key=lambda x: abs(x.get("home_win_prob", 0.5) - 0.5), reverse=True)

        # Print
        print(f"\n  {'AWAY':<25} {'HOME':<25} {'PICK':<20} {'PROB':>6} {'SPREAD':>8}")
        print("  " + "-" * 86)
        for p in predictions:
            prob = max(p["home_win_prob"], p["away_win_prob"])
            winner = p.get("predicted_winner", "?")
            spread = p.get("model_spread", 0)
            print(f"  {p['away_team']:<25} {p['home_team']:<25} "
                  f"{winner:<20} {prob:>5.1%} {spread:>+7.1f}")

        # Save
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        out_path = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")

        return predictions

    def calculate_edges(self, predictions: list, market_odds: dict) -> list:
        """Calculate model edge vs market odds."""
        for pred in predictions:
            home = pred["home_team"]
            away = pred["away_team"]
            game_key = f"{away} @ {home}"

            market = market_odds.get(game_key, {})
            if market:
                market_prob = market.get("home_implied_prob", 0.5)
                model_prob = pred["home_win_prob"]
                edge = model_prob - market_prob

                pred["market_ml_home"] = market.get("home_ml", "N/A")
                pred["market_ml_away"] = market.get("away_ml", "N/A")
                pred["edge"] = round(edge, 4)
                pred["edge_pct"] = f"{edge:.1%}"

                if abs(edge) >= 0.08:
                    pred["signal"] = "STRONG BET"
                elif abs(edge) >= 0.04:
                    pred["signal"] = "MODEL HIGHER" if edge > 0 else "MARKET HIGHER"
                else:
                    pred["signal"] = "AGREE"

        return predictions

    def _fetch_today_schedule(self) -> list:
        """Fetch today's schedule from ESPN API."""
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            games = []
            for event in data.get("events", []):
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                if len(competitors) < 2:
                    continue

                home = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away = next((c for c in competitors if c.get("homeAway") == "away"), {})

                games.append({
                    "game_id": event.get("id", ""),
                    "home_team": home.get("team", {}).get("displayName", "Unknown"),
                    "away_team": away.get("team", {}).get("displayName", "Unknown"),
                    "game_time": event.get("date", ""),
                })

            return games
        except Exception as e:
            print(f"  [WARN] Could not fetch schedule: {e}")
            return []


if __name__ == "__main__":
    dp = DailyPredictor()
    dp.predict_today()
