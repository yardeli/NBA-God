"""
NBA-God — Flask Web Server
============================
Dashboard on port 5052.
Endpoints: /, /api/data, /api/odds, /api/daily, /api/paperbets, /api/predict/<home>/<away>
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__,
            template_folder=str(ROOT / "web" / "templates"),
            static_folder=str(ROOT / "web" / "static"))

PORT = int(os.environ.get("NBA_GOD_PORT", 5052))

_predictor = None

# ESPN sometimes abbreviates team names — normalize for matching
_TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
}

def _normalize_team(name):
    """Normalize team name for consistent matching."""
    low = name.strip().lower()
    return _TEAM_ALIASES.get(low, low)


def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from daily_predictor import DailyPredictor
            _predictor = DailyPredictor()
            _predictor._load()
        except Exception as e:
            print(f"[Server] Could not load predictor: {e}")
    return _predictor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    today = datetime.now().strftime("%Y%m%d")
    preds_file = ROOT / "outputs" / f"predictions_{today}.json"
    bets_file = ROOT / "outputs" / f"bets_log_{today}.json"
    importance_file = ROOT / "phase5_deploy" / "output" / "feature_importance.json"

    predictions = []
    if preds_file.exists():
        with open(preds_file) as f:
            predictions = json.load(f)

    bets = {}
    if bets_file.exists():
        with open(bets_file) as f:
            bets = json.load(f)

    importance = {}
    if importance_file.exists():
        with open(importance_file) as f:
            importance = json.load(f)

    return jsonify({
        "predictions": predictions,
        "bets": bets,
        "importance": importance,
        "date": datetime.now().strftime("%B %d, %Y"),
        "n_games": len(predictions),
    })


@app.route("/api/odds")
def api_odds():
    try:
        from odds_fetcher import fetch_championship_odds
        return jsonify(fetch_championship_odds())
    except Exception as e:
        return jsonify({"error": str(e), "teams": []})


@app.route("/api/daily")
def api_daily():
    """Today's NBA games with model predictions and betting edges."""
    from datetime import timezone

    predictor = get_predictor()
    if predictor is None:
        return jsonify({"games": [], "error": "Model not loaded", "meta": {}})

    try:
        predictions = predictor.predict_today()
    except Exception as e:
        return jsonify({"games": [], "error": str(e), "meta": {}})

    # Fetch market odds (keyed by normalized lowercase names for robust matching)
    market_odds = {}
    try:
        from odds_fetcher import fetch_todays_games, format_american
        market_data = fetch_todays_games(force_refresh=True)
        if market_data and not market_data.get("error"):
            for g in market_data.get("games", []):
                away_n = _normalize_team(g.get('away_team', ''))
                home_n = _normalize_team(g.get('home_team', ''))
                key = f"{away_n} @ {home_n}"
                market_odds[key] = g
    except Exception:
        pass

    # Also check ESPN for live game status + scores
    live_games = set()
    live_scores = {}  # team_name_lower -> {score, period, clock}
    try:
        import requests as req
        espn = req.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            timeout=8).json()
        now = datetime.now(timezone.utc)
        for ev in espn.get("events", []):
            status_obj = ev.get("status", {})
            status = status_obj.get("type", {}).get("name", "")
            period = status_obj.get("period", 0)
            clock = status_obj.get("displayClock", "")
            if status == "STATUS_IN_PROGRESS":
                comps = ev.get("competitions", [{}])[0].get("competitors", [])
                for c in comps:
                    tname = c.get("team", {}).get("displayName", "")
                    norm = _normalize_team(tname)
                    live_games.add(norm)
                    live_scores[norm] = {
                        "score": int(c.get("score", 0)),
                        "period": period,
                        "clock": clock,
                    }
    except Exception:
        pass

    def _fmt_american(ml):
        if ml is None:
            return "--"
        ml = int(ml)
        return f"+{ml}" if ml > 0 else str(ml)

    def _implied(ml):
        if ml is None:
            return None
        ml = int(ml)
        if ml > 0:
            return 100 / (ml + 100)
        return abs(ml) / (abs(ml) + 100)

    enriched_games = []
    for pred in predictions:
        home = pred.get("home_team", "Unknown")
        away = pred.get("away_team", "Unknown")
        home_prob = pred.get("home_win_prob", 0.5)
        away_prob = pred.get("away_win_prob", 0.5)
        spread = pred.get("model_spread", 0)

        # Check if game is live (normalize for ESPN abbreviations)
        in_progress = (_normalize_team(home) in live_games or _normalize_team(away) in live_games)

        # Match to market odds (use normalized names for matching)
        game_key = f"{_normalize_team(away)} @ {_normalize_team(home)}"
        mkt = market_odds.get(game_key, {})

        # Build bets array
        bets = []
        for side, team, prob in [("home", home, home_prob), ("away", away, away_prob)]:
            # Model odds (American format)
            if 0 < prob < 1:
                if prob >= 0.5:
                    model_american = f"-{round((prob / (1 - prob)) * 100)}"
                else:
                    model_american = f"+{round(((1 - prob) / prob) * 100)}"
            else:
                model_american = "--"

            # Pre-game market odds
            mkt_ml = mkt.get(f"{side}_ml")
            mkt_implied = _implied(mkt_ml)

            # Edge vs pre-game line (positive = model likes this team MORE than market)
            edge = (prob - mkt_implied) if mkt_implied else None
            signal = ""
            if edge is not None:
                if edge >= 0.08:
                    signal = "STRONG VALUE"      # Model loves this team, market undervalues
                elif edge >= 0.04:
                    signal = "MODEL HIGHER"      # Model slightly higher than market
                elif edge <= -0.08:
                    signal = "FADE"              # Market overvalues this team, avoid
                elif edge <= -0.04:
                    signal = "MARKET HIGHER"     # Market slightly higher than model
                else:
                    signal = "AGREE"             # Model and market roughly aligned

            bets.append({
                "bet": f"ML {team}",
                "bet_type": "moneyline",
                "side": side,
                "team": team,
                "model_prob": round(prob, 4),
                "model_odds": model_american,
                "market_odds": _fmt_american(mkt_ml),
                "market_odds_raw": int(mkt_ml) if mkt_ml else None,
                "market_implied": round(mkt_implied, 4) if mkt_implied else None,
                "edge": round(edge, 4) if edge is not None else None,
                "signal": signal,
                "market_spread": mkt.get(f"{side}_spread") if side == "home" else mkt.get("away_spread"),
                "total": mkt.get("total"),
            })

        n_value = sum(1 for b in bets if b["signal"] in ("STRONG VALUE", "MODEL HIGHER"))
        best_edge = max((abs(b["edge"]) for b in bets if b["edge"] is not None), default=0)

        # Live score info
        home_score_info = live_scores.get(_normalize_team(home), {})
        away_score_info = live_scores.get(_normalize_team(away), {})

        enriched_games.append({
            "game_id": pred.get("game_id", f"{away}_{home}"),
            "home_team": home,
            "away_team": away,
            "home_display": home,
            "away_display": away,
            "home_win_prob": home_prob,
            "away_win_prob": away_prob,
            "in_progress": in_progress,
            "home_score": home_score_info.get("score"),
            "away_score": away_score_info.get("score"),
            "period": home_score_info.get("period") or away_score_info.get("period"),
            "clock": home_score_info.get("clock") or away_score_info.get("clock", ""),
            "market_spread": mkt.get("home_spread"),
            "market_total": mkt.get("total"),
            "prediction": {
                "prob_home_wins": home_prob,
                "prob_away_wins": away_prob,
                "model_spread": spread,
                "confidence": pred.get("confidence", "medium"),
            },
            "bets": bets,
            "n_value_bets": n_value,
            "best_edge": best_edge,
        })

    return jsonify({
        "games": enriched_games,
        "meta": {
            "date": datetime.now().strftime("%B %d, %Y"),
            "n_games": len(enriched_games),
        },
    })


@app.route("/api/paperbets")
def api_paperbets():
    today = datetime.now().strftime("%Y%m%d")
    bets_file = ROOT / "outputs" / f"bets_log_{today}.json"
    if bets_file.exists():
        with open(bets_file) as f:
            return jsonify(json.load(f))
    return jsonify({"total_bets": 0, "wins": 0, "losses": 0, "profit": 0})


@app.route("/api/predict/<home>/<away>")
def api_predict(home, away):
    predictor = get_predictor()
    if predictor is None:
        return jsonify({"error": "Model not loaded"})
    try:
        result = predictor.predict_game(home, away)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print(f"\n  NBA-God Server starting on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)
