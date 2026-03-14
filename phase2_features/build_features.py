"""
NBA-God — Phase 2: Feature Engineering for Multi-Era Data
===========================================================
Builds a clean, leak-free feature matrix from the Phase 1 database.

NBA-specific features (Four Factors + efficiency metrics):
  Tier 1 — Universal (1946+): win%, point differential, home/away, rest_days, momentum, back_to_back
  Tier 2 — Box score (1980+): eFG%, TO rate, ORB rate, FT rate, pace, efficiency ratings
  Tier 3 — Advanced (2000+): net efficiency, 3pt rate, assist/turnover ratio

CRITICAL: NO DATA LEAKAGE — every feature uses shift(1) / pre-game data only.

OUTPUT:
  phase2_features/output/features_all.parquet
  phase2_features/output/features_t1.parquet
  phase2_features/output/features_t2.parquet
  phase2_features/output/feature_docs.json
  phase2_features/output/missingness_report.json
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "phase1_data" / "output" / "nba_god.db"
OUT_DIR = ROOT / "phase2_features" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ROLLING_WINDOW = 10      # Games for rolling stats (82-game season)
MOMENTUM_WINDOW = 5      # Recent games for momentum
MIN_GAMES_HISTORY = 3    # Minimum prior games needed
RANDOM_SEED = 42

# ── Feature Documentation ────────────────────────────────────────────────────
FEATURE_DOCS = {
    # Tier 1 — Universal
    "win_pct":        {"tier": 1, "era": "all", "desc": "Season-to-date win% before this game"},
    "avg_margin":     {"tier": 1, "era": "all", "desc": "Average point margin season-to-date"},
    "pts_pg":         {"tier": 1, "era": "all", "desc": "Points per game season-to-date"},
    "opp_pts_pg":     {"tier": 1, "era": "all", "desc": "Opponent points per game season-to-date"},
    "games_played":   {"tier": 1, "era": "all", "desc": "Games played before this game"},
    "rest_days":      {"tier": 1, "era": "all", "desc": "Days since last game (capped at 10)"},
    "back_to_back":   {"tier": 1, "era": "all", "desc": "1 if playing second game in 2 days"},
    "win_streak":     {"tier": 1, "era": "all", "desc": "Current win streak (neg = losing)"},
    "momentum":       {"tier": 1, "era": "all", "desc": "Win% over last 5 games"},
    "home_away":      {"tier": 1, "era": "all", "desc": "1 if home team, 0 if away"},
    "h2h_win_pct":    {"tier": 1, "era": "all", "desc": "H2H win% in last 10 meetings"},
    "h2h_games":      {"tier": 1, "era": "all", "desc": "Total prior H2H games"},
    # Tier 2 — Four Factors + box score
    "efg_pct":        {"tier": 2, "era": "1980+", "desc": "Effective FG% (rolling)"},
    "to_rate":        {"tier": 2, "era": "1980+", "desc": "Turnover rate (rolling)"},
    "orb_rate":       {"tier": 2, "era": "1980+", "desc": "Offensive rebound rate (rolling)"},
    "ft_rate":        {"tier": 2, "era": "1980+", "desc": "Free throw rate FTM/FGA (rolling)"},
    "pace":           {"tier": 2, "era": "1980+", "desc": "Estimated possessions per game (rolling)"},
    "off_eff":        {"tier": 2, "era": "1980+", "desc": "Offensive efficiency pts/100pos (rolling)"},
    "def_eff":        {"tier": 2, "era": "1980+", "desc": "Defensive efficiency opp_pts/100pos (rolling)"},
    # Tier 3 — Advanced
    "net_eff":        {"tier": 3, "era": "2000+", "desc": "Net efficiency (off - def)"},
    "3pt_rate":       {"tier": 3, "era": "2000+", "desc": "3PA / FGA ratio (rolling)"},
    "ast_to":         {"tier": 3, "era": "2000+", "desc": "Assist-to-turnover ratio (rolling)"},
    "pythagorean_wp": {"tier": 3, "era": "2000+", "desc": "Pythagorean win% (exponent=13.91 for NBA)"},
}

np.random.seed(RANDOM_SEED)


# ── Load data ────────────────────────────────────────────────────────────────

def load_data(conn: sqlite3.Connection):
    print("Loading games from database...")
    # Query only columns that exist in the database
    cursor = conn.execute("PRAGMA table_info(games)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    core_cols = [
        "game_id", "season", "game_date", "game_type",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name",
        "home_pts", "away_pts", "home_win",
        "margin", "total_pts",
        "era", "data_completeness_tier",
    ]
    optional_cols = [
        "overtime_periods",
        "home_fgm", "home_fga", "away_fgm", "away_fga",
        "home_fg3m", "home_fg3a", "away_fg3m", "away_fg3a",
        "home_ftm", "home_fta", "away_ftm", "away_fta",
        "home_oreb", "home_dreb", "home_reb",
        "away_oreb", "away_dreb", "away_reb",
        "home_ast", "home_stl", "home_blk", "home_tov",
        "away_ast", "away_stl", "away_blk", "away_tov",
        "home_pf", "away_pf",
    ]

    select_cols = [c for c in core_cols if c in existing_cols]
    select_cols += [c for c in optional_cols if c in existing_cols]

    sql = f"SELECT {', '.join(select_cols)} FROM games ORDER BY game_date"
    games = pd.read_sql(sql, conn)
    print(f"  {len(games):,} games loaded ({len(select_cols)} columns).")
    return games


# ── Tier 1: Universal rolling team stats ─────────────────────────────────────

def build_team_season_stats(games: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team, game), compute cumulative stats BEFORE this game.
    Explode home/away into per-team rows, then vectorized cumsum.
    """
    print("  Building cumulative season stats (Tier 1)...")

    home = games[["game_id", "season", "game_date", "home_team_id", "away_team_id",
                   "home_pts", "away_pts", "home_win"]].copy()
    home["team_id"] = home["home_team_id"]
    home["opp_id"] = home["away_team_id"]
    home["pts"] = home["home_pts"]
    home["opp_pts"] = home["away_pts"]
    home["won"] = home["home_win"]
    home["is_home"] = 1

    away = games[["game_id", "season", "game_date", "home_team_id", "away_team_id",
                   "home_pts", "away_pts", "home_win"]].copy()
    away["team_id"] = away["away_team_id"]
    away["opp_id"] = away["home_team_id"]
    away["pts"] = away["away_pts"]
    away["opp_pts"] = away["home_pts"]
    away["won"] = 1 - away["home_win"]
    away["is_home"] = 0

    tg = pd.concat([
        home[["game_id", "season", "game_date", "team_id", "opp_id",
              "pts", "opp_pts", "won", "is_home"]],
        away[["game_id", "season", "game_date", "team_id", "opp_id",
              "pts", "opp_pts", "won", "is_home"]],
    ], ignore_index=True)
    tg = tg.sort_values(["team_id", "season", "game_date"]).reset_index(drop=True)
    tg["margin"] = tg["pts"] - tg["opp_pts"]

    grp = tg.groupby(["team_id", "season"])

    # Games played before this game (0-indexed)
    tg["games_played"] = grp.cumcount()

    # Cumulative wins, points before this game
    tg["cum_wins"] = grp["won"].cumsum() - tg["won"]
    tg["win_pct"] = tg["cum_wins"] / tg["games_played"].replace(0, np.nan)

    tg["cum_pts"] = grp["pts"].cumsum() - tg["pts"]
    tg["pts_pg"] = tg["cum_pts"] / tg["games_played"].replace(0, np.nan)

    tg["cum_opp_pts"] = grp["opp_pts"].cumsum() - tg["opp_pts"]
    tg["opp_pts_pg"] = tg["cum_opp_pts"] / tg["games_played"].replace(0, np.nan)

    tg["cum_margin"] = grp["margin"].cumsum() - tg["margin"]
    tg["avg_margin"] = tg["cum_margin"] / tg["games_played"].replace(0, np.nan)

    # Rest days and back-to-back detection
    tg["game_date_dt"] = pd.to_datetime(tg["game_date"])
    tg["prev_date"] = grp["game_date_dt"].shift(1)
    tg["rest_days"] = (tg["game_date_dt"] - tg["prev_date"]).dt.days.clip(upper=10)
    tg["back_to_back"] = (tg["rest_days"] == 1).astype(int)

    # Win streak
    print("    Computing win streaks...")
    def win_streak_vec(won_arr):
        streaks = np.zeros(len(won_arr), dtype=int)
        s = 0
        for i, w in enumerate(won_arr):
            streaks[i] = s
            s = (max(s, 0) + 1) if w == 1 else (min(s, 0) - 1)
        return streaks

    streak_vals = []
    for (_, _), sub in tg.groupby(["team_id", "season"]):
        streak_vals.append(win_streak_vec(sub["won"].values))
    tg["win_streak"] = np.concatenate(streak_vals)

    # Momentum (rolling win% over last MOMENTUM_WINDOW games)
    print("    Computing momentum...")
    tg["momentum"] = (
        grp["won"].transform(
            lambda x: x.rolling(window=MOMENTUM_WINDOW, min_periods=3).mean().shift(1)
        )
    )

    # Pythagorean win% (NBA exponent ~13.91 per Morey)
    tg["pyth_num"] = tg["cum_pts"] ** 13.91
    tg["pyth_den"] = tg["cum_pts"] ** 13.91 + tg["cum_opp_pts"] ** 13.91
    tg["pythagorean_wp"] = tg["pyth_num"] / tg["pyth_den"].replace(0, np.nan)

    keep = ["game_id", "team_id", "is_home", "games_played", "win_pct",
            "avg_margin", "pts_pg", "opp_pts_pg", "rest_days", "back_to_back",
            "win_streak", "momentum", "pythagorean_wp"]
    return tg[keep].copy()


# ── Tier 2: Rolling Four Factors + efficiency stats ──────────────────────────

def build_rolling_box_stats(games: pd.DataFrame,
                            window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Rolling {window}-game averages of Four Factors and efficiency metrics.
    Computed strictly before each game (shift(1)).
    """
    print(f"  Building rolling {window}-game box score stats (Tier 2)...")

    rows = []
    for _, g in games.iterrows():
        for side in ["home", "away"]:
            fgm = g.get(f"{side}_fgm")
            fga = g.get(f"{side}_fga")
            fg3m = g.get(f"{side}_fg3m")
            fg3a = g.get(f"{side}_fg3a")
            ftm = g.get(f"{side}_ftm")
            fta = g.get(f"{side}_fta")
            oreb = g.get(f"{side}_oreb")
            dreb = g.get(f"{side}_dreb")
            ast = g.get(f"{side}_ast")
            tov = g.get(f"{side}_tov")
            pts = g.get(f"{side}_pts")

            opp = "away" if side == "home" else "home"
            opp_pts = g.get(f"{opp}_pts")
            opp_fga = g.get(f"{opp}_fga")
            opp_fta = g.get(f"{opp}_fta")
            opp_tov = g.get(f"{opp}_tov")
            opp_oreb = g.get(f"{opp}_oreb")
            opp_dreb = g.get(f"{opp}_dreb")

            # eFG% = (FGM + 0.5 * FG3M) / FGA
            efg = None
            if fga and fga > 0 and fgm is not None:
                efg = (fgm + 0.5 * (fg3m or 0)) / fga

            # Turnover rate = TOV / (FGA + 0.44*FTA + TOV)
            to_rate = None
            if tov is not None and fga and fga > 0:
                possessions_proxy = fga + 0.44 * (fta or 0) + (tov or 0)
                if possessions_proxy > 0:
                    to_rate = (tov or 0) / possessions_proxy

            # ORB rate = OREB / (OREB + opp_DREB)
            orb_rate = None
            if oreb is not None and opp_dreb is not None:
                denom = (oreb or 0) + (opp_dreb or 0)
                if denom > 0:
                    orb_rate = (oreb or 0) / denom

            # FT rate = FTM / FGA
            ft_rate = None
            if ftm is not None and fga and fga > 0:
                ft_rate = ftm / fga

            # Pace estimate = FGA + 0.44*FTA - OREB + TOV
            pace = None
            if fga is not None and tov is not None:
                pace = (fga or 0) + 0.44 * (fta or 0) - (oreb or 0) + (tov or 0)

            # Offensive efficiency = pts / possessions * 100
            off_eff = None
            if pace and pace > 0 and pts is not None:
                off_eff = (pts / pace) * 100

            # Defensive efficiency = opp_pts / opp_possessions * 100
            def_eff = None
            if opp_pts is not None and opp_fga is not None and opp_tov is not None:
                opp_pace = (opp_fga or 0) + 0.44 * (opp_fta or 0) - (opp_oreb or 0) + (opp_tov or 0)
                if opp_pace > 0:
                    def_eff = (opp_pts / opp_pace) * 100

            # 3pt rate = FG3A / FGA
            three_rate = None
            if fg3a is not None and fga and fga > 0:
                three_rate = fg3a / fga

            # AST/TOV ratio
            ast_to = None
            if ast is not None and tov and tov > 0:
                ast_to = ast / tov

            rows.append({
                "game_id": g["game_id"],
                "team_id": g[f"{side}_team_id"],
                "season": g["season"],
                "game_date": g["game_date"],
                "efg": efg,
                "to_rate": to_rate,
                "orb_rate": orb_rate,
                "ft_rate": ft_rate,
                "pace": pace,
                "off_eff": off_eff,
                "def_eff": def_eff,
                "three_rate": three_rate,
                "ast_to": ast_to,
            })

    box_df = pd.DataFrame(rows)
    box_df = box_df.sort_values(["team_id", "season", "game_date"]).reset_index(drop=True)

    # Rolling averages
    min_p = max(1, window // 3)
    grp = box_df.groupby(["team_id", "season"])

    for col, out_col in [
        ("efg", "efg_pct_rolling"),
        ("to_rate", "to_rate_rolling"),
        ("orb_rate", "orb_rate_rolling"),
        ("ft_rate", "ft_rate_rolling"),
        ("pace", "pace_rolling"),
        ("off_eff", "off_eff_rolling"),
        ("def_eff", "def_eff_rolling"),
        ("three_rate", "3pt_rate_rolling"),
        ("ast_to", "ast_to_rolling"),
    ]:
        box_df[out_col] = grp[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_p).mean().shift(1)
        )

    # Net efficiency
    box_df["net_eff_rolling"] = box_df["off_eff_rolling"] - box_df["def_eff_rolling"]

    keep = ["game_id", "team_id", "efg_pct_rolling", "to_rate_rolling",
            "orb_rate_rolling", "ft_rate_rolling", "pace_rolling",
            "off_eff_rolling", "def_eff_rolling", "net_eff_rolling",
            "3pt_rate_rolling", "ast_to_rolling"]
    return box_df[keep].copy()


# ── Head-to-head history ─────────────────────────────────────────────────────

def build_h2h_features(games: pd.DataFrame) -> pd.DataFrame:
    """H2H win% in last 10 meetings between each pair of teams."""
    print("  Building head-to-head features...")

    home = games[["game_id", "season", "game_date",
                   "home_team_id", "away_team_id", "home_win"]].copy()
    home["team_id"] = home["home_team_id"]
    home["opp_id"] = home["away_team_id"]
    home["won"] = home["home_win"]

    away = games[["game_id", "season", "game_date",
                   "home_team_id", "away_team_id", "home_win"]].copy()
    away["team_id"] = away["away_team_id"]
    away["opp_id"] = away["home_team_id"]
    away["won"] = 1 - away["home_win"]

    tg = pd.concat([home[["game_id", "game_date", "team_id", "opp_id", "won"]],
                     away[["game_id", "game_date", "team_id", "opp_id", "won"]]],
                    ignore_index=True)
    tg = tg.sort_values("game_date").reset_index(drop=True)

    h2h_results = []
    pair_history: dict[tuple, list] = {}

    for _, row in tg.iterrows():
        pair = tuple(sorted([row["team_id"], row["opp_id"]]))
        history = pair_history.get(pair, [])

        if history:
            last_10 = history[-10:]
            team_wins = sum(1 for h in last_10 if h["winner"] == row["team_id"])
            h2h_wp = team_wins / len(last_10)
            h2h_n = len(history)
        else:
            h2h_wp = np.nan
            h2h_n = 0

        h2h_results.append({
            "game_id": row["game_id"],
            "team_id": row["team_id"],
            "h2h_win_pct": h2h_wp,
            "h2h_games": h2h_n,
        })

        winner = row["team_id"] if row["won"] == 1 else row["opp_id"]
        if pair not in pair_history:
            pair_history[pair] = []
        pair_history[pair].append({"winner": winner})

    return pd.DataFrame(h2h_results)


# ── Assemble matchup features ───────────────────────────────────────────────

def assemble_matchup_features(games: pd.DataFrame,
                               team_stats: pd.DataFrame,
                               box_stats: pd.DataFrame,
                               h2h: pd.DataFrame) -> pd.DataFrame:
    """
    Create matchup feature matrix with team1 vs team2 differentials.
    Random team1/team2 assignment prevents leakage.
    """
    print("  Assembling matchup features...")

    np.random.seed(RANDOM_SEED)
    flip = np.random.rand(len(games)) > 0.5
    df = games.copy()

    df["team1_id"] = np.where(flip, df["home_team_id"], df["away_team_id"])
    df["team2_id"] = np.where(flip, df["away_team_id"], df["home_team_id"])
    df["team1_is_home"] = np.where(flip, 1, 0)
    df["label"] = np.where(
        flip,
        df["home_win"],
        1 - df["home_win"]
    )

    # Merge team stats for team1 and team2
    for side, team_col in [("t1", "team1_id"), ("t2", "team2_id")]:
        ts = team_stats.copy()
        stat_cols = [c for c in ts.columns if c not in ("game_id", "team_id", "is_home")]
        ts = ts.rename(columns={c: f"{side}_{c}" for c in stat_cols})
        ts = ts.rename(columns={"team_id": team_col})
        df = df.merge(ts[["game_id", team_col] + [f"{side}_{c}" for c in stat_cols]],
                       on=["game_id", team_col], how="left")

    # Merge box stats
    for side, team_col in [("t1", "team1_id"), ("t2", "team2_id")]:
        bs = box_stats.copy()
        box_cols = [c for c in bs.columns if c not in ("game_id", "team_id")]
        bs = bs.rename(columns={c: f"{side}_{c}" for c in box_cols})
        bs = bs.rename(columns={"team_id": team_col})
        df = df.merge(bs[["game_id", team_col] + [f"{side}_{c}" for c in box_cols]],
                       on=["game_id", team_col], how="left")

    # Merge H2H
    for side, team_col in [("t1", "team1_id"), ("t2", "team2_id")]:
        hh = h2h.copy()
        hh_cols = ["h2h_win_pct", "h2h_games"]
        hh = hh.rename(columns={c: f"{side}_{c}" for c in hh_cols})
        hh = hh.rename(columns={"team_id": team_col})
        df = df.merge(hh[["game_id", team_col] + [f"{side}_{c}" for c in hh_cols]],
                       on=["game_id", team_col], how="left")

    # Build differential columns
    diff_pairs = {
        "win_pct":       ("t1_win_pct",           "t2_win_pct"),
        "avg_margin":    ("t1_avg_margin",         "t2_avg_margin"),
        "pts_pg":        ("t1_pts_pg",             "t2_pts_pg"),
        "opp_pts_pg":    ("t1_opp_pts_pg",         "t2_opp_pts_pg"),
        "rest_days":     ("t1_rest_days",           "t2_rest_days"),
        "back_to_back":  ("t1_back_to_back",        "t2_back_to_back"),
        "win_streak":    ("t1_win_streak",          "t2_win_streak"),
        "momentum":      ("t1_momentum",            "t2_momentum"),
        "h2h_win_pct":   ("t1_h2h_win_pct",        "t2_h2h_win_pct"),
        "pythagorean_wp": ("t1_pythagorean_wp",     "t2_pythagorean_wp"),
        "efg_pct":       ("t1_efg_pct_rolling",     "t2_efg_pct_rolling"),
        "to_rate":       ("t1_to_rate_rolling",      "t2_to_rate_rolling"),
        "orb_rate":      ("t1_orb_rate_rolling",     "t2_orb_rate_rolling"),
        "ft_rate":       ("t1_ft_rate_rolling",      "t2_ft_rate_rolling"),
        "pace":          ("t1_pace_rolling",         "t2_pace_rolling"),
        "off_eff":       ("t1_off_eff_rolling",      "t2_off_eff_rolling"),
        "def_eff":       ("t1_def_eff_rolling",      "t2_def_eff_rolling"),
        "net_eff":       ("t1_net_eff_rolling",      "t2_net_eff_rolling"),
        "3pt_rate":      ("t1_3pt_rate_rolling",     "t2_3pt_rate_rolling"),
        "ast_to":        ("t1_ast_to_rolling",       "t2_ast_to_rolling"),
    }

    print("    Computing differentials (team1 - team2)...")
    for feat_name, (col1, col2) in diff_pairs.items():
        if col1 in df.columns and col2 in df.columns:
            df[f"diff_{feat_name}"] = df[col1] - df[col2]
        else:
            df[f"diff_{feat_name}"] = np.nan

    # Home/away indicator (team1 perspective)
    df["home_away"] = df["team1_is_home"]

    # H2H games (team-agnostic)
    df["h2h_games"] = df.get("t1_h2h_games", 0)

    return df


# ── Era normalization ────────────────────────────────────────────────────────

def normalize_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score diff_* features within each season."""
    print("  Normalizing features by season (z-score)...")
    diff_cols = [c for c in df.columns if c.startswith("diff_")]

    for col in diff_cols:
        mean = df.groupby("season")[col].transform("mean")
        std = df.groupby("season")[col].transform("std")
        df[f"{col}_z"] = (df[col] - mean) / std.replace(0, np.nan)

    return df


# ── Reporting ────────────────────────────────────────────────────────────────

def generate_feature_report(df: pd.DataFrame):
    print("\n  Generating feature report...")
    diff_cols = [c for c in df.columns if c.startswith("diff_")]

    overall_miss = {col: round(df[col].isna().mean() * 100, 1) for col in diff_cols}
    n_all = len(df)
    n_tier2 = df["diff_efg_pct"].notna().sum()

    report = {
        "total_games": n_all,
        "tier1_games": n_all,
        "tier2_games": int(n_tier2),
        "features": FEATURE_DOCS,
        "missingness_overall": overall_miss,
        "games_by_era": df["era"].value_counts().to_dict(),
        "label_balance": round(df["label"].mean(), 4),
    }

    with open(OUT_DIR / "feature_docs.json", "w") as f:
        json.dump(FEATURE_DOCS, f, indent=2)
    with open(OUT_DIR / "missingness_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  PHASE 2 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"\n  Total games: {n_all:,}")
    print(f"  Tier 2 (box stats): {n_tier2:,}")
    print(f"  Diff features: {len(diff_cols)}")
    print(f"  Label balance: {report['label_balance'] * 100:.1f}% (should ~50%)")
    print(f"\n  Reports saved to: {OUT_DIR}")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nNBA-God — Phase 2: Feature Engineering")
    print(f"DB: {DB_PATH}\n")

    conn = sqlite3.connect(DB_PATH)
    try:
        games = load_data(conn)
    finally:
        conn.close()

    print("\n[1/5] Building cumulative season stats (Tier 1)...")
    team_stats = build_team_season_stats(games)

    print("\n[2/5] Building rolling box stats (Tier 2)...")
    box_stats = build_rolling_box_stats(games)

    print("\n[3/5] Building H2H features...")
    h2h_feats = build_h2h_features(games)

    print("\n[4/5] Assembling matchup features...")
    features = assemble_matchup_features(games, team_stats, box_stats, h2h_feats)

    print("\n[5/5] Normalizing and saving...")
    features = normalize_by_season(features)
    features = features[features["diff_win_pct"].notna()].copy()
    print(f"  After filtering: {len(features):,} games")

    features.to_parquet(OUT_DIR / "features_all.parquet", index=False)
    print(f"  Saved: features_all.parquet")

    # Tier 1 only
    t1_cols = [c for c in features.columns
               if not any(x in c for x in ["efg_pct", "to_rate_rolling", "orb_rate",
                                             "ft_rate_rolling", "pace_rolling",
                                             "off_eff", "def_eff", "net_eff",
                                             "3pt_rate", "ast_to"])]
    t1 = features[t1_cols].dropna(subset=["diff_win_pct", "diff_avg_margin"])
    t1.to_parquet(OUT_DIR / "features_t1.parquet", index=False)
    print(f"  Saved: features_t1.parquet ({len(t1):,} games)")

    # Tier 1+2
    t2 = features[features["diff_efg_pct"].notna()].copy()
    t2.to_parquet(OUT_DIR / "features_t2.parquet", index=False)
    print(f"  Saved: features_t2.parquet ({len(t2):,} games)")

    generate_feature_report(features)


if __name__ == "__main__":
    main()
