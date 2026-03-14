"""
NBA-God — Phase 3: Era-Aware Model Training
=============================================
Four approaches for handling 80 years of basketball data with different rules,
scoring levels, and data availability.

APPROACH A — Time-decayed weighting (recent seasons matter more)
APPROACH B — Era-stratified ensemble (separate models per era)
APPROACH C — Single model with era features
APPROACH D — Transfer learning (pre-train historical, fine-tune modern)

VALIDATION — Walk-forward CPCV with 1-season embargo.
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (log_loss, brier_score_loss, accuracy_score,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FEAT_DIR = ROOT / "phase2_features" / "output"
OUT_DIR = ROOT / "phase3_models" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
TEST_SEASONS = list(range(2015, 2026))
EMBARGO_SEASONS = 1
MIN_TRAIN_SEASONS = 5

DECAY_HALFLIVES = [5, 10, 15, 20]

XGB_BASE = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)


def get_feature_cols(df: pd.DataFrame, tier: int = 3) -> list:
    t1 = [c for c in df.columns if c.startswith("diff_") and
          not any(x in c for x in ["efg_pct", "to_rate", "orb_rate",
                                     "ft_rate", "pace", "off_eff", "def_eff",
                                     "net_eff", "3pt_rate", "ast_to",
                                     "pythagorean"])]
    t2 = [c for c in df.columns if c.startswith("diff_")]
    extra = ["home_away", "h2h_games"]

    if tier == 1:
        return [c for c in t1 + extra if c in df.columns]
    elif tier == 2:
        return [c for c in t2 + extra if c in df.columns]
    else:
        all_feats = t2 + extra
        z_cols = [c for c in df.columns if c.endswith("_z")]
        return list(dict.fromkeys([c for c in all_feats + z_cols if c in df.columns]))


def cpcv_splits(df, test_seasons, embargo=1):
    for test_year in test_seasons:
        max_train = test_year - embargo - 1
        if max_train < df["season"].min() + MIN_TRAIN_SEASONS:
            continue
        train_idx = df[df["season"] <= max_train].index
        test_idx = df[df["season"] == test_year].index
        if len(train_idx) < 500 or len(test_idx) < 50:
            continue
        yield train_idx, test_idx, test_year


def evaluate(y_true, y_prob, season):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "season": season,
        "n_games": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "auc": round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else np.nan,
    }


# ── Approach A: Time-Decayed ─────────────────────────────────────────────────

def approach_a(df, half_life):
    print(f"\n  [A] Half-life={half_life}y")
    results = []
    feat_cols = get_feature_cols(df, tier=3)
    current_year = df["season"].max()

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train, test = df.loc[train_idx], df.loc[test_idx]
        X_train, y_train = train[feat_cols].fillna(0), train["label"].values
        X_test, y_test = test[feat_cols].fillna(0), test["label"].values

        weights = np.power(0.5, (current_year - train["season"]) / half_life).values

        val_mask = train["season"] == train["season"].max()
        model = xgb.XGBClassifier(**XGB_BASE)
        model.fit(X_train[~val_mask], y_train[~val_mask],
                  sample_weight=weights[~val_mask],
                  eval_set=[(X_train[val_mask], y_train[val_mask])],
                  verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, y_prob, test_year)
        m["approach"] = f"A_hl{half_life}"
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f} n={m['n_games']}")
    return results


# ── Approach B: Era-Stratified ───────────────────────────────────────────────

def approach_b(df):
    print(f"\n  [B] Era-stratified ensemble")
    results = []
    feat_a = get_feature_cols(df, tier=1)
    feat_b = get_feature_cols(df, tier=2)
    feat_c = get_feature_cols(df, tier=3)

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train, test = df.loc[train_idx], df.loc[test_idx]
        X_test_c = test[feat_c].fillna(0)
        y_test = test["label"].values

        def train_sub(era_df, feats):
            X = era_df[feats].fillna(0)
            y = era_df["label"].values
            if len(X) < 100:
                return None
            val_mask = era_df["season"] == era_df["season"].max()
            m = xgb.XGBClassifier(**{**XGB_BASE, "n_estimators": 300})
            try:
                m.fit(X[~val_mask], y[~val_mask],
                      eval_set=[(X[val_mask], y[val_mask])], verbose=False)
            except Exception:
                m.fit(X, y, verbose=False)
            return m

        m_a = train_sub(train[train["season"] <= 1994], feat_a)
        m_b = train_sub(train[(train["season"] >= 1995) & (train["season"] <= 2015)], feat_b)
        m_c = train_sub(train[train["season"] >= 2016], feat_c)

        def pred(model, feats, df_):
            if model is None:
                return np.full(len(df_), 0.5)
            return model.predict_proba(df_[feats].fillna(0))[:, 1]

        p_a = pred(m_a, feat_a, test)
        p_b = pred(m_b, feat_b, test)
        p_c = pred(m_c, feat_c, test)

        y_prob = 0.2 * p_a + 0.3 * p_b + 0.5 * p_c
        m = evaluate(y_test, y_prob, test_year)
        m["approach"] = "B"
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f} n={m['n_games']}")
    return results


# ── Approach C: Single + Era Features ────────────────────────────────────────

def approach_c(df):
    print(f"\n  [C] Single model + era features")
    results = []
    feat_cols = get_feature_cols(df, tier=3)

    df = df.copy()
    df["season_numeric"] = df["season"].astype(float)
    feat_cols = feat_cols + ["season_numeric", "data_completeness_tier"]
    feat_cols = list(dict.fromkeys([c for c in feat_cols if c in df.columns]))

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train, test = df.loc[train_idx], df.loc[test_idx]
        X_train, y_train = train[feat_cols].fillna(0), train["label"].values
        X_test, y_test = test[feat_cols].fillna(0), test["label"].values

        val_mask = train["season"] == train["season"].max()
        model = xgb.XGBClassifier(**XGB_BASE)
        model.fit(X_train[~val_mask], y_train[~val_mask],
                  eval_set=[(X_train[val_mask], y_train[val_mask])],
                  verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, y_prob, test_year)
        m["approach"] = "C"
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f} n={m['n_games']}")
    return results


# ── Approach D: Transfer Learning ────────────────────────────────────────────

def approach_d(df):
    print(f"\n  [D] Transfer learning")
    results = []
    feat_cols = get_feature_cols(df, tier=3)
    PRETRAIN_CUTOFF = 2012

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train, test = df.loc[train_idx], df.loc[test_idx]
        X_test, y_test = test[feat_cols].fillna(0), test["label"].values

        pretrain = train[train["season"] <= PRETRAIN_CUTOFF]
        finetune = train[train["season"] > PRETRAIN_CUTOFF]
        if len(pretrain) < 500 or len(finetune) < 200:
            continue

        X_pre, y_pre = pretrain[feat_cols].fillna(0), pretrain["label"].values
        val_p = pretrain["season"] == pretrain["season"].max()
        base = xgb.XGBClassifier(**{**XGB_BASE, "n_estimators": 300})
        try:
            base.fit(X_pre[~val_p], y_pre[~val_p],
                     eval_set=[(X_pre[val_p], y_pre[val_p])], verbose=False)
        except Exception:
            base.fit(X_pre, y_pre, verbose=False)

        X_ft, y_ft = finetune[feat_cols].fillna(0), finetune["label"].values
        val_f = finetune["season"] == finetune["season"].max()
        fine = xgb.XGBClassifier(**{**XGB_BASE, "n_estimators": 200, "learning_rate": 0.02})
        try:
            fine.fit(X_ft[~val_f], y_ft[~val_f],
                     eval_set=[(X_ft[val_f], y_ft[val_f])],
                     xgb_model=base.get_booster(), verbose=False)
        except Exception:
            fine.fit(X_ft, y_ft, verbose=False)

        y_prob = fine.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, y_prob, test_year)
        m["approach"] = "D_transfer"
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f} n={m['n_games']}")
    return results


# ── Production model ─────────────────────────────────────────────────────────

def train_production_model(df, best_approach="C"):
    print(f"\nTraining production model (Approach {best_approach}, all data)...")
    feat_cols = get_feature_cols(df, tier=3)

    if best_approach == "C":
        df = df.copy()
        df["season_numeric"] = df["season"].astype(float)
        feat_cols = feat_cols + ["season_numeric", "data_completeness_tier"]
        feat_cols = list(dict.fromkeys([c for c in feat_cols if c in df.columns]))

    X = df[feat_cols].fillna(0)
    y = df["label"].values

    prod_params = {**XGB_BASE, "early_stopping_rounds": None}
    model = xgb.XGBClassifier(**prod_params)
    model.fit(X, y, verbose=False)

    # Also train logistic regression for ensemble
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    lr.fit(X_scaled, y)

    pkg = {
        "xgb_model": model,
        "lr_model": lr,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "approach": best_approach,
        "trained_on": f"{df['season'].min()}-{df['season'].max()}",
        "n_games": len(df),
        "ensemble_weights": {"xgb": 0.65, "lr": 0.35},
    }
    model_path = OUT_DIR / "production_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  Production model saved: {model_path}")
    return model


# ── Comparison report ────────────────────────────────────────────────────────

def build_comparison_report(all_results):
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(OUT_DIR / "all_results_raw.csv", index=False)

    agg = (rdf.groupby("approach").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_log_loss=("log_loss", "mean"),
        avg_brier=("brier", "mean"),
        avg_auc=("auc", "mean"),
        seasons_tested=("season", "count"),
    ).round(4).reset_index())
    agg = agg.sort_values("avg_accuracy", ascending=False)

    print("\n" + "=" * 70)
    print("  PHASE 3 MODEL COMPARISON")
    print("=" * 70)
    print(f"\n  {'Approach':<18} {'Accuracy':>9} {'Log-Loss':>9} {'Brier':>8} {'AUC':>7}")
    print("  " + "-" * 55)
    for _, r in agg.iterrows():
        print(f"  {r['approach']:<18} {r['avg_accuracy']:>9.4f} "
              f"{r['avg_log_loss']:>9.4f} {r['avg_brier']:>8.4f} {r['avg_auc']:>7.4f}")

    summary = {
        "comparison": agg.to_dict(orient="records"),
        "best_overall": agg.iloc[0]["approach"],
    }
    with open(OUT_DIR / "comparison_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Best: {summary['best_overall']}")
    print("=" * 70)
    return summary


def compute_feature_importance(df):
    feat_cols = get_feature_cols(df, tier=3)
    X = df[feat_cols].fillna(0)
    y = df["label"].values

    model = xgb.XGBClassifier(**{**XGB_BASE, "early_stopping_rounds": None,
                                   "n_estimators": 300})
    model.fit(X, y, verbose=False)

    importance = dict(zip(feat_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    top20 = dict(list(importance.items())[:20])

    print("\n  Top 20 features:")
    for feat, imp in top20.items():
        bar = "#" * int(imp * 500)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    with open(OUT_DIR / "feature_importance.json", "w") as f:
        json.dump({"top20": {k: float(v) for k, v in top20.items()}}, f, indent=2)
    return top20


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nNBA-God — Phase 3: Era-Aware Model Training")

    df = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    df = df[df["diff_win_pct"].notna()].copy()
    df = df[df["season"] < 2026].copy()

    print(f"  {len(df):,} games, {df['season'].min()}-{df['season'].max()}")
    print(f"  Features: {len(get_feature_cols(df, 3))}")

    all_results = []

    print("=" * 60)
    print("APPROACH A: Time-decayed")
    for hl in DECAY_HALFLIVES:
        all_results.extend(approach_a(df, hl))

    print("=" * 60)
    print("APPROACH B: Era-stratified")
    all_results.extend(approach_b(df))

    print("=" * 60)
    print("APPROACH C: Single + era features")
    all_results.extend(approach_c(df))

    print("=" * 60)
    print("APPROACH D: Transfer learning")
    all_results.extend(approach_d(df))

    summary = build_comparison_report(all_results)
    compute_feature_importance(df)
    train_production_model(df, summary["best_overall"])

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
