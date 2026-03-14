"""
NBA-God — CLI Prediction Interface
====================================
Usage:
  python predict.py --today              # Predict today's NBA games
  python predict.py --team1 "Lakers" --team2 "Celtics"  # Head-to-head
  python predict.py --pipeline           # Run full pipeline
  python predict.py --backtest           # Walk-forward backtest
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config


def run_full_pipeline(force_scrape=False):
    """Full pipeline: ingest -> features -> train -> deploy."""
    print("\n" + "=" * 60)
    print("  NBA-GOD -- FULL PIPELINE")
    print("=" * 60)

    # Phase 1
    print("\n[Phase 1] Data Ingestion...")
    from phase1_data.ingest import main as ingest_main
    ingest_main()

    # Elo ratings
    print("\n[Elo] Building ratings...")
    import sqlite3
    import pandas as pd
    from elo import build_elo_ratings
    conn = sqlite3.connect(ROOT / "phase1_data" / "output" / "nba_god.db")
    games = pd.read_sql("SELECT * FROM games ORDER BY game_date", conn)
    conn.close()
    elo_system, games_with_elo = build_elo_ratings(games)

    # Phase 2
    print("\n[Phase 2] Feature Engineering...")
    from phase2_features.build_features import main as features_main
    features_main()

    # Phase 3
    print("\n[Phase 3] Model Training...")
    from phase3_models.train import main as train_main
    train_main()

    # Phase 4
    print("\n[Phase 4] Playoff Simulation...")
    from phase4_playoffs.playoff_sim import main as playoff_main
    playoff_main()

    # Phase 5
    print("\n[Phase 5] Robustness & Deployment...")
    from phase5_deploy.robustness import main as deploy_main
    deploy_main()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


def predict_today():
    """Predict today's NBA games."""
    from daily_predictor import DailyPredictor
    dp = DailyPredictor()
    predictions = dp.predict_today()

    # Try to add market odds
    try:
        from odds_fetcher import fetch_todays_games
        market = fetch_todays_games()
        if market.get("games") and not market.get("error"):
            dp.calculate_edges(predictions, market)
    except Exception:
        pass

    return predictions


def predict_matchup(team1, team2, neutral=False):
    """Predict a specific matchup."""
    from daily_predictor import DailyPredictor
    dp = DailyPredictor()
    result = dp.predict_game(team1, team2)

    print(f"\n  {result['away_team']} @ {result['home_team']}")
    print(f"  Prediction: {result['predicted_winner']} "
          f"({max(result['home_win_prob'], result['away_win_prob']):.1%})")
    print(f"  Point spread: {result.get('model_point_spread', 'N/A')}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA-God -- Predict any NBA game")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline")
    parser.add_argument("--today", action="store_true", help="Predict today's games")
    parser.add_argument("--team1", type=str, help="Home team")
    parser.add_argument("--team2", type=str, help="Away team")
    parser.add_argument("--neutral", action="store_true", help="Neutral site")
    parser.add_argument("--backtest", action="store_true", help="Walk-forward backtest")

    args = parser.parse_args()

    if args.pipeline:
        run_full_pipeline()
    elif args.today:
        predict_today()
    elif args.team1 and args.team2:
        predict_matchup(args.team1, args.team2, args.neutral)
    elif args.backtest:
        print("Run: python phase3_models/train.py")
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python predict.py --pipeline')
        print('  python predict.py --today')
        print('  python predict.py --team1 "Lakers" --team2 "Celtics"')
