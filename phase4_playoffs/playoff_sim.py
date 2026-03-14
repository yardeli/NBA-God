"""
NBA-God — Phase 4: Playoff & NBA Finals Simulation
====================================================
NBA playoff structure (2021+, with Play-In):
  - Play-In Tournament: 7-10 seeds play for final 2 playoff spots per conference
  - First Round: Best-of-7 (1v8, 2v7, 3v6, 4v5)
  - Conference Semifinals: Best-of-7
  - Conference Finals: Best-of-7
  - NBA Finals: Best-of-7

Monte Carlo simulation: 10,000 runs -> championship probabilities.
"""

import json
import pickle
import sqlite3
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).parent.parent
FEAT_DIR = ROOT / "phase2_features" / "output"
P3_DIR = ROOT / "phase3_models" / "output"
OUT_DIR = ROOT / "phase4_playoffs" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Playoff round metadata ──────────────────────────────────────────────────
PLAYOFF_ROUNDS = [
    {"name": "Play-In", "format": "single_elim", "games": 1},
    {"name": "First Round", "format": "best_of_7", "games": 7},
    {"name": "Conference Semifinals", "format": "best_of_7", "games": 7},
    {"name": "Conference Finals", "format": "best_of_7", "games": 7},
    {"name": "NBA Finals", "format": "best_of_7", "games": 7},
]

# NBA home court advantage pattern in best-of-7: 2-2-1-1-1
# Higher seed gets games 1,2,5,7 at home


def load_base_model():
    with open(P3_DIR / "production_model.pkl", "rb") as f:
        pkg = pickle.load(f)
    return pkg


def simulate_series(p_higher_seed: float, series_format: str,
                    rng: np.random.Generator) -> bool:
    """
    Simulate a playoff series.
    Returns True if higher seed wins.
    """
    if series_format == "single_elim":
        return rng.random() < p_higher_seed

    # Best-of-7
    wins_needed = 4
    h_wins, l_wins = 0, 0
    game_num = 0
    # Home court pattern: games 0,1,4,6 at home for higher seed (2-2-1-1-1)
    home_games = {0, 1, 4, 6}

    while h_wins < wins_needed and l_wins < wins_needed:
        # Slight home court bump for the team with home court
        is_home = game_num in home_games
        if is_home:
            game_p = min(p_higher_seed + 0.03, 0.95)  # Small home boost
        else:
            game_p = max(p_higher_seed - 0.03, 0.05)

        if rng.random() < game_p:
            h_wins += 1
        else:
            l_wins += 1
        game_num += 1

    return h_wins >= wins_needed


def get_matchup_probability(team_a: int, team_b: int,
                             features_df: pd.DataFrame,
                             model_pkg: dict) -> float:
    """Get P(team_a beats team_b) from the model."""
    xgb_model = model_pkg["xgb_model"]
    feat_cols = model_pkg["feature_cols"]

    mask = (
        ((features_df["team1_id"] == team_a) & (features_df["team2_id"] == team_b)) |
        ((features_df["team1_id"] == team_b) & (features_df["team2_id"] == team_a))
    )
    if mask.any():
        row = features_df[mask].iloc[-1]
        flipped = (row["team1_id"] == team_b)
        avail = [c for c in feat_cols if c in row.index]
        X = pd.DataFrame([row[avail].fillna(0)]).reindex(columns=feat_cols, fill_value=0)
        prob = float(xgb_model.predict_proba(X)[0, 1])
        return (1 - prob) if flipped else prob

    return 0.5


def simulate_play_in(seeds: dict, features_df: pd.DataFrame,
                     model_pkg: dict, rng: np.random.Generator) -> dict:
    """
    Simulate NBA Play-In Tournament.
    Seeds 7-10 compete for the 7th and 8th playoff spots.
    7 vs 8 -> winner is 7 seed, loser plays winner of 9v10
    9 vs 10 -> loser eliminated, winner plays loser of 7v8
    """
    t7 = seeds.get("7")
    t8 = seeds.get("8")
    t9 = seeds.get("9")
    t10 = seeds.get("10")

    if not all([t7, t8, t9, t10]):
        return {"7": t7 or seeds.get("7"), "8": t8 or seeds.get("8")}

    # 7 vs 8: winner clinches 7 seed
    p78 = get_matchup_probability(t7, t8, features_df, model_pkg)
    if rng.random() < p78:
        seed_7_winner = t7
        loser_78 = t8
    else:
        seed_7_winner = t8
        loser_78 = t7

    # 9 vs 10: loser eliminated
    p910 = get_matchup_probability(t9, t10, features_df, model_pkg)
    if rng.random() < p910:
        winner_910 = t9
    else:
        winner_910 = t10

    # Loser of 7/8 vs Winner of 9/10 for 8 seed
    p_final = get_matchup_probability(loser_78, winner_910, features_df, model_pkg)
    if rng.random() < p_final:
        seed_8_winner = loser_78
    else:
        seed_8_winner = winner_910

    return {"7": seed_7_winner, "8": seed_8_winner}


def simulate_bracket(season: int, playoff_teams: dict,
                     features_df: pd.DataFrame,
                     model_pkg: dict,
                     n_simulations: int = 10_000) -> dict:
    """
    Simulate the NBA postseason bracket.

    playoff_teams: {
        "East": {"1": team_id, "2": team_id, ..., "10": team_id},
        "West": {"1": team_id, "2": team_id, ..., "10": team_id},
    }
    """
    print(f"  Simulating {n_simulations:,} postseasons for {season}...")

    rng = np.random.default_rng(42)
    champion_counts = defaultdict(int)
    finals_appearances = defaultdict(int)

    for sim in range(n_simulations):
        conf_winners = {}

        for conf in ["East", "West"]:
            seeds = playoff_teams.get(conf, {})
            if len(seeds) < 6:
                continue

            # Play-In Tournament (seeds 7-10)
            play_in_results = simulate_play_in(seeds, features_df, model_pkg, rng)
            final_seeds = {k: v for k, v in seeds.items() if int(k) <= 6}
            final_seeds["7"] = play_in_results.get("7", seeds.get("7"))
            final_seeds["8"] = play_in_results.get("8", seeds.get("8"))

            # First Round: 1v8, 4v5, 3v6, 2v7
            first_round = [
                (final_seeds.get("1"), final_seeds.get("8")),
                (final_seeds.get("4"), final_seeds.get("5")),
                (final_seeds.get("3"), final_seeds.get("6")),
                (final_seeds.get("2"), final_seeds.get("7")),
            ]

            round2 = []
            for high, low in first_round:
                if high and low:
                    p = get_matchup_probability(high, low, features_df, model_pkg)
                    if simulate_series(p, "best_of_7", rng):
                        round2.append(high)
                    else:
                        round2.append(low)
                elif high:
                    round2.append(high)

            # Conference Semifinals: bracket stays intact
            semis = []
            for i in range(0, len(round2) - 1, 2):
                t1, t2 = round2[i], round2[i + 1]
                p = get_matchup_probability(t1, t2, features_df, model_pkg)
                if simulate_series(p, "best_of_7", rng):
                    semis.append(t1)
                else:
                    semis.append(t2)

            # Conference Finals
            if len(semis) == 2:
                p = get_matchup_probability(semis[0], semis[1], features_df, model_pkg)
                if simulate_series(p, "best_of_7", rng):
                    conf_winners[conf] = semis[0]
                else:
                    conf_winners[conf] = semis[1]

        # NBA Finals
        east_champ = conf_winners.get("East")
        west_champ = conf_winners.get("West")
        if east_champ and west_champ:
            finals_appearances[east_champ] += 1
            finals_appearances[west_champ] += 1
            p = get_matchup_probability(east_champ, west_champ, features_df, model_pkg)
            if simulate_series(p, "best_of_7", rng):
                champion_counts[east_champ] += 1
            else:
                champion_counts[west_champ] += 1

    # Build results
    champ_probs = sorted(
        [(tid, cnt / n_simulations) for tid, cnt in champion_counts.items()],
        key=lambda x: -x[1]
    )
    finals_probs = sorted(
        [(tid, cnt / n_simulations) for tid, cnt in finals_appearances.items()],
        key=lambda x: -x[1]
    )

    return {
        "season": season,
        "n_simulations": n_simulations,
        "champion_probabilities": champ_probs[:16],
        "finals_appearance": finals_probs[:16],
    }


# ── Postseason game calibration ──────────────────────────────────────────────

def calibrate_postseason(features_df: pd.DataFrame, model_pkg: dict) -> dict:
    """Train calibration layer for postseason games."""
    print("  Calibrating postseason predictions...")

    post_types = {"first_round", "conference_semis", "conference_finals", "nba_finals", "play_in"}
    post_df = features_df[features_df["game_type"].isin(post_types)].copy()

    if len(post_df) < 50:
        print("    Insufficient postseason data for calibration.")
        return {}

    feat_cols = model_pkg["feature_cols"]
    xgb_model = model_pkg["xgb_model"]

    avail = [c for c in feat_cols if c in post_df.columns]
    X = post_df[avail].fillna(0).reindex(columns=feat_cols, fill_value=0)
    y = post_df["label"].values

    probs = xgb_model.predict_proba(X)[:, 1]
    base_acc = accuracy_score(y, (probs >= 0.5).astype(int))

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, y)
    cal_probs = cal.predict(probs)
    cal_acc = accuracy_score(y, (cal_probs >= 0.5).astype(int))

    print(f"    Postseason: base_acc={base_acc:.3f}  cal_acc={cal_acc:.3f}  n={len(post_df)}")

    return {"calibrator": cal, "base_acc": base_acc, "cal_acc": cal_acc}


# ── Historical postseason analysis ──────────────────────────────────────────

def analyze_postseason_history(features_df: pd.DataFrame) -> dict:
    """Analyze historical postseason patterns."""
    print("\n  Analyzing postseason history...")

    post_types = {"first_round", "conference_semis", "conference_finals", "nba_finals"}
    post_df = features_df[features_df["game_type"].isin(post_types)].copy()

    if len(post_df) == 0:
        return {}

    by_round = {}
    for gt in post_types:
        sub = post_df[post_df["game_type"] == gt]
        if len(sub) > 0:
            by_round[gt] = {
                "games": len(sub),
                "home_win_rate": round(sub["home_away"].corr(sub["label"]), 3)
                    if "home_away" in sub.columns else None,
            }

    return {"by_round": by_round, "total_postseason_games": len(post_df)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nNBA-God — Phase 4: Playoff Simulation\n")

    model_pkg = load_base_model()
    features = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    features = features[features["season"] < 2026].copy()

    print("[1/3] Postseason calibration...")
    cal_result = calibrate_postseason(features, model_pkg)

    print("\n[2/3] Historical postseason analysis...")
    history = analyze_postseason_history(features)
    with open(OUT_DIR / "postseason_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print("\n[3/3] Bracket simulation...")
    db_path = ROOT / "phase1_data" / "output" / "nba_god.db"
    team_names = {}
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        teams = pd.read_sql("SELECT team_id, canonical_name FROM teams", conn)
        conn.close()
        team_names = dict(zip(teams["team_id"], teams["canonical_name"]))

    # Save playoff model
    pkg = {
        "base_model": model_pkg,
        "calibration": cal_result,
        "postseason_history": history,
    }
    with open(OUT_DIR / "playoff_model.pkl", "wb") as f:
        pickle.dump(pkg, f)

    print("\n" + "=" * 60)
    print("  PHASE 4 SUMMARY")
    print("=" * 60)
    if cal_result:
        print(f"  Postseason calibration: base={cal_result.get('base_acc', 'N/A'):.3f}  "
              f"cal={cal_result.get('cal_acc', 'N/A'):.3f}")
    print(f"  All outputs saved to: {OUT_DIR}")
    print("=" * 60)
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
