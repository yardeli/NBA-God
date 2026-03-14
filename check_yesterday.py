"""
NBA-God — Check Yesterday's Results
=====================================
Retroactive P&L check with colored terminal output.
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).parent

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def fetch_yesterday_scores():
    """Fetch yesterday's final scores from ESPN."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yesterday}"

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        scores = {}
        for event in data.get("events", []):
            comps = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            home_name = home.get("team", {}).get("displayName", "")
            away_name = away.get("team", {}).get("displayName", "")
            key = f"{away_name} @ {home_name}"
            scores[key] = {
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "home_team": home_name,
                "away_team": away_name,
            }
        return scores, yesterday
    except Exception as e:
        print(f"Error fetching scores: {e}")
        return {}, yesterday


def check_results():
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    preds_file = ROOT / "outputs" / f"predictions_{yesterday}.json"

    if not preds_file.exists():
        print(f"No predictions file for {yesterday}")
        return

    with open(preds_file) as f:
        preds = json.load(f)

    scores, date_str = fetch_yesterday_scores()

    print(f"\n{BOLD}{'=' * 70}")
    print(f"  NBA-GOD RESULTS -- {datetime.strptime(date_str, '%Y%m%d').strftime('%B %d, %Y')}")
    print(f"{'=' * 70}{RESET}\n")

    correct, total, strong_correct, strong_total = 0, 0, 0, 0
    profit = 0
    stake = 100

    for pred in preds:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        key = f"{away} @ {home}"
        score = scores.get(key)

        if not score:
            # Try reverse key
            for sk, sv in scores.items():
                if home.lower() in sk.lower() and away.lower() in sk.lower():
                    score = sv
                    break

        if not score:
            continue

        winner_pred = pred.get("predicted_winner", "")
        home_won = score["home_score"] > score["away_score"]
        actual_winner = score["home_team"] if home_won else score["away_team"]

        is_correct = (winner_pred.lower() in actual_winner.lower() or
                      actual_winner.lower() in winner_pred.lower())

        prob = max(pred.get("home_win_prob", 0.5), pred.get("away_win_prob", 0.5))
        signal = pred.get("signal", "")

        total += 1
        if is_correct:
            correct += 1
            color = GREEN
            result = "WIN"
            profit += stake * 0.91
        else:
            color = RED
            result = "LOSS"
            profit -= stake

        is_strong = prob >= 0.58 or signal == "STRONG BET"
        if is_strong:
            strong_total += 1
            if is_correct:
                strong_correct += 1

        score_str = f"{score['away_score']}-{score['home_score']}"
        print(f"  {color}{result:>4}{RESET}  "
              f"{away:<22} @ {home:<22} "
              f"Score: {score_str:>7}  "
              f"Pick: {winner_pred:<20} ({prob:.0%})"
              f"{'  **STRONG**' if is_strong else ''}")

    print(f"\n{'=' * 70}")
    if total > 0:
        pct = correct / total * 100
        color = GREEN if pct >= 55 else (YELLOW if pct >= 50 else RED)
        print(f"  {BOLD}Overall:{RESET} {color}{correct}/{total} ({pct:.1f}%){RESET}")
        print(f"  {BOLD}Profit:{RESET} {GREEN if profit > 0 else RED}${profit:+.0f}{RESET}")

    if strong_total > 0:
        spct = strong_correct / strong_total * 100
        color = GREEN if spct >= 55 else RED
        print(f"  {BOLD}Strong bets:{RESET} {color}{strong_correct}/{strong_total} ({spct:.1f}%){RESET}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    check_results()
