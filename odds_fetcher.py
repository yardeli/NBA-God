"""
NBA-God — Live Odds Fetcher
=============================
Pulls NBA game odds from The Odds API.
Sport key: "basketball_nba"
Markets: moneyline (h2h), point spreads (spreads), totals (over/under)
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
CACHE_FILE = Path(__file__).parent / "phase5_deploy" / "output" / "odds_cache.json"
CACHE_TTL = 3600
DAILY_CACHE_FILE = Path(__file__).parent / "phase5_deploy" / "output" / "daily_odds_cache.json"
DAILY_CACHE_TTL = 300


def american_to_implied(american: int) -> float:
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def implied_to_american(prob: float) -> str:
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        return f"{int(-(prob / (1 - prob)) * 100)}"
    else:
        return f"+{int(((1 - prob) / prob) * 100)}"


def format_american(american: int) -> str:
    return f"+{american}" if american > 0 else str(american)


def remove_vig(probs: list[float]) -> list[float]:
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else probs


def _get_api_key() -> Optional[str]:
    key = os.environ.get("ODDS_API_KEY") or os.environ.get("BETTING_ODS_API")
    if not key or key == "your_api_key_here":
        return None
    return key


def _error_response(msg):
    return {"teams": [], "games": [], "error": msg, "cached": False,
            "fetched_at": time.time(), "requests_remaining": -1}


def _load_cache_or_error(cache_file, msg):
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["cached"] = True
            cached["error"] = f"[stale] {msg}"
            return cached
        except Exception:
            pass
    return _error_response(msg)


def fetch_championship_odds(force_refresh=False):
    """Fetch NBA Championship futures."""
    if not force_refresh and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if time.time() - cached.get("fetched_at", 0) < CACHE_TTL:
                cached["cached"] = True
                return cached
        except Exception:
            pass

    api_key = _get_api_key()
    if not api_key:
        return _error_response("ODDS_API_KEY not set")

    try:
        params = {"apiKey": api_key, "regions": "us",
                  "markets": "outrights", "oddsFormat": "american"}
        resp = requests.get(f"{ODDS_API_BASE}/sports/{SPORT}/odds/",
                            params=params, timeout=10)

        if resp.status_code == 422 or (resp.status_code == 200 and not resp.json()):
            # Outrights not available — try alternate sport keys
            alt_sports = [
                "basketball_nba_championship_winner",
                "basketball_nba",
            ]
            for alt in alt_sports:
                alt_resp = requests.get(
                    f"{ODDS_API_BASE}/sports/{alt}/odds/",
                    params={**params, "markets": "h2h"}, timeout=10)
                if alt_resp.status_code == 200 and alt_resp.json():
                    resp = alt_resp
                    break
            else:
                return _load_cache_or_error(
                    CACHE_FILE, "NBA futures market not available right now")

        if resp.status_code != 200:
            return _load_cache_or_error(CACHE_FILE, f"API {resp.status_code}")

        data = resp.json()
        remaining = int(resp.headers.get("x-requests-remaining", -1))

        team_odds = {}
        for event in data:
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") not in ("outrights", "h2h"):
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price")
                        if name and price is not None:
                            team_odds.setdefault(name, []).append(
                                (int(price), book.get("title", ""))
                            )

        teams = []
        for name, prices in team_odds.items():
            best_price, best_book = max(prices, key=lambda x: x[0])
            implied = american_to_implied(best_price)
            teams.append({
                "name": name, "best_odds": best_price,
                "best_odds_fmt": format_american(best_price),
                "best_book": best_book, "implied_prob": round(implied, 4),
            })

        raw_probs = [t["implied_prob"] for t in teams]
        fair_probs = remove_vig(raw_probs)
        for t, fp in zip(teams, fair_probs):
            t["fair_prob"] = round(fp, 4)
        teams.sort(key=lambda x: -x["fair_prob"])

        result = {"teams": teams, "requests_remaining": remaining,
                   "error": None, "cached": False, "fetched_at": time.time()}
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(result, indent=2))
        return result

    except Exception as e:
        return _load_cache_or_error(CACHE_FILE, str(e))


def fetch_todays_games(force_refresh=False):
    """Fetch moneyline, point spread, and totals for today's NBA games."""
    if not force_refresh and DAILY_CACHE_FILE.exists():
        try:
            cached = json.loads(DAILY_CACHE_FILE.read_text(encoding="utf-8"))
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if (time.time() - cached.get("fetched_at", 0) < DAILY_CACHE_TTL and
                    cached.get("date") == today_str):
                cached["cached"] = True
                return cached
        except Exception:
            pass

    api_key = _get_api_key()
    if not api_key:
        return _error_response("ODDS_API_KEY not set")

    try:
        resp = requests.get(f"{ODDS_API_BASE}/sports/{SPORT}/odds/",
                            params={"apiKey": api_key, "regions": "us",
                                    "markets": "h2h,spreads,totals",
                                    "oddsFormat": "american"},
                            timeout=10)
        if resp.status_code != 200:
            return _load_cache_or_error(DAILY_CACHE_FILE, f"API {resp.status_code}")

        data = resp.json()
        remaining = int(resp.headers.get("x-requests-remaining", -1))
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        games = []
        for event in data:
            ct = event.get("commence_time", "")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")

            # Parse best moneyline, spread, and total from bookmakers
            home_mls, away_mls = [], []
            home_spreads, away_spreads = [], []
            totals_over = []

            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    mkey = market.get("key")
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price")
                        name = outcome.get("name", "")
                        point = outcome.get("point")
                        if price is None:
                            continue
                        if mkey == "h2h":
                            if name == home_team:
                                home_mls.append(price)
                            elif name == away_team:
                                away_mls.append(price)
                        elif mkey == "spreads":
                            if name == home_team:
                                home_spreads.append((point, price))
                            elif name == away_team:
                                away_spreads.append((point, price))
                        elif mkey == "totals" and name == "Over" and point:
                            totals_over.append(point)

            # Use consensus (median-ish: pick the middle bookmaker)
            home_ml = sorted(home_mls)[len(home_mls) // 2] if home_mls else None
            away_ml = sorted(away_mls)[len(away_mls) // 2] if away_mls else None
            home_spread = sorted(home_spreads)[len(home_spreads) // 2][0] if home_spreads else None
            total = sorted(totals_over)[len(totals_over) // 2] if totals_over else None

            games.append({
                "id": event.get("id"),
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": ct,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_spread": home_spread,
                "away_spread": -home_spread if home_spread is not None else None,
                "total": total,
                "n_books": len(event.get("bookmakers", [])),
            })

        result = {"games": games, "date": today_str, "n_games": len(games),
                   "requests_remaining": remaining, "cached": False,
                   "error": None, "fetched_at": time.time()}

        DAILY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAILY_CACHE_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    except Exception as e:
        return _load_cache_or_error(DAILY_CACHE_FILE, str(e))


if __name__ == "__main__":
    print("Fetching NBA odds...")
    futures = fetch_championship_odds()
    if futures.get("error"):
        print(f"  Error: {futures['error']}")
    else:
        print(f"  NBA Championship futures ({len(futures['teams'])} teams):")
        for t in futures["teams"][:10]:
            print(f"    {t['name']:<30} {t['best_odds_fmt']:>8}  ({t['fair_prob']:.1%})")

    daily = fetch_todays_games()
    if daily.get("error"):
        print(f"  Daily error: {daily['error']}")
    else:
        print(f"\n  Today's games: {daily['n_games']}")
        for g in daily["games"][:5]:
            print(f"    {g['away_team']} @ {g['home_team']}")
