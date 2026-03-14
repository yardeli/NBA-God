"""
NBA-God — Dashboard Runner
============================
Launches the web dashboard with today's predictions, P&L tracking,
and NBA Championship futures.

Port: 5052 (NBA)
"""

import json
import os
import sys
import time
import signal
import subprocess
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PORT = 5052
SERVER_URL = f"http://localhost:{PORT}"


def fetch_scores() -> dict:
    """Fetch today's scores from ESPN API."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
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
            status = comps.get("status", {}).get("type", {}).get("name", "")
            key = f"{away_name} @ {home_name}"
            scores[key] = {
                "home_score": int(home.get("score", 0) or 0),
                "away_score": int(away.get("score", 0) or 0),
                "status": status,
                "home_team": home_name,
                "away_team": away_name,
            }
        return scores
    except Exception as e:
        print(f"  [WARN] Score fetch failed: {e}")
        return {}


def compute_pnl(predictions_file: str, scores: dict) -> dict:
    """Compute paper P&L from predictions vs actual results."""
    try:
        with open(predictions_file) as f:
            preds = json.load(f)
    except Exception:
        return {"total_bets": 0, "wins": 0, "losses": 0, "pending": 0,
                "profit": 0, "roi": 0, "bets": []}

    bets = []
    wins, losses, pending = 0, 0, 0
    stake = 100  # $100 per bet

    for pred in preds:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        key = f"{away} @ {home}"
        score = scores.get(key, {})

        prob = max(pred.get("home_win_prob", 0.5), pred.get("away_win_prob", 0.5))
        signal = pred.get("signal", "")

        # Only bet on strong signals
        if prob < 0.55 and signal != "STRONG BET":
            continue

        winner = pred.get("predicted_winner", "")
        actual_status = score.get("status", "")

        bet = {
            "game": key,
            "pick": winner,
            "prob": prob,
            "signal": signal,
            "stake": stake,
        }

        if actual_status == "STATUS_FINAL":
            home_won = score["home_score"] > score["away_score"]
            actual_winner = score["home_team"] if home_won else score["away_team"]
            bet["result"] = "WIN" if actual_winner == winner else "LOSS"
            bet["actual_winner"] = actual_winner
            bet["score"] = f"{score['away_score']}-{score['home_score']}"

            if bet["result"] == "WIN":
                bet["profit"] = round(stake * 0.91, 2)  # -110 odds
                wins += 1
            else:
                bet["profit"] = -stake
                losses += 1
        else:
            bet["result"] = "PENDING"
            bet["profit"] = 0
            pending += 1

        bets.append(bet)

    total_profit = sum(b["profit"] for b in bets)
    total_staked = sum(b["stake"] for b in bets if b["result"] != "PENDING")

    return {
        "total_bets": len(bets),
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        "bets": bets,
    }


def save_bets_log(pnl: dict):
    """Save bets to log file."""
    log_dir = ROOT / "outputs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"bets_log_{datetime.now().strftime('%Y%m%d')}.json"
    with open(log_file, "w") as f:
        json.dump(pnl, f, indent=2, default=str)


def kill_old_server():
    """Kill any existing server on our port."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f":{PORT}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
        else:
            subprocess.run(["fuser", "-k", f"{PORT}/tcp"],
                           capture_output=True)
    except Exception:
        pass


def start_server():
    """Start the Flask web server."""
    kill_old_server()
    time.sleep(0.5)

    server_script = ROOT / "web" / "server.py"
    env = os.environ.copy()
    env["NBA_GOD_PORT"] = str(PORT)

    proc = subprocess.Popen(
        [sys.executable, str(server_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    return proc


def open_browser():
    """Open dashboard in browser."""
    webbrowser.open(SERVER_URL)


def main():
    print("\n" + "=" * 60)
    print("  NBA-GOD DASHBOARD")
    print("=" * 60)

    # Generate predictions if not already done today
    preds_file = ROOT / "outputs" / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    if not preds_file.exists():
        print("\n  Generating today's predictions...")
        from daily_predictor import DailyPredictor
        dp = DailyPredictor()
        dp.predict_today()

    # Start server
    print(f"\n  Starting server on port {PORT}...")
    proc = start_server()

    # Open browser
    print(f"  Opening {SERVER_URL}...")
    open_browser()

    print(f"\n  Dashboard running at {SERVER_URL}")
    print("  Press Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
