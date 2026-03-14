"""
NBA-God — Phase 5: Robustness & Deployment Readiness
======================================================
1. Bootstrap CI — 1000 resamples on 2015-2025 test window
2. Worst-case analysis — by season, game type, era
3. Calibration audit — reliability diagram + ECE/MCE
4. Feature importance — SHAP or XGBoost fscore
5. Prediction interface — NBAGodPredictor class
6. Final report
"""

import json
import pickle
import sqlite3
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_artifacts():
    print("Loading artifacts...")
    with open(ROOT / "phase3_models" / "output" / "production_model.pkl", "rb") as f:
        pkg = pickle.load(f)

    features = pd.read_parquet(ROOT / "phase2_features" / "output" / "features_all.parquet")

    db_path = ROOT / "phase1_data" / "output" / "nba_god.db"
    team_names = {}
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        teams = pd.read_sql("SELECT team_id, canonical_name FROM teams", conn)
        conn.close()
        team_names = dict(zip(teams["team_id"], teams["canonical_name"]))

    return pkg, features, team_names


# ── 1. Bootstrap CI ──────────────────────────────────────────────────────────

def bootstrap_ci(features, model_pkg, test_seasons=range(2015, 2026),
                 n_boot=1000, seed=42):
    print("\n[1/6] Bootstrap confidence intervals...")

    test_df = features[features["season"].isin(test_seasons)].copy()
    if len(test_df) == 0:
        return {}

    feat_cols = model_pkg["feature_cols"]
    xgb_model = model_pkg["xgb_model"]

    avail = [c for c in feat_cols if c in test_df.columns]
    X = test_df[avail].fillna(0).reindex(columns=feat_cols, fill_value=0)
    y = test_df["label"].values
    probs = xgb_model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    rng = np.random.default_rng(seed)
    n = len(y)
    metrics = defaultdict(list)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi, pi, pr = y[idx], preds[idx], probs[idx]
        metrics["acc"].append(float((yi == pi).mean()))
        metrics["logloss"].append(float(log_loss(yi, np.clip(pr, 1e-6, 1 - 1e-6))))
        metrics["brier"].append(float(brier_score_loss(yi, pr)))

    def ci(vals):
        a = np.array(vals)
        return {"mean": float(a.mean()), "ci_lo": float(np.percentile(a, 2.5)),
                "ci_hi": float(np.percentile(a, 97.5))}

    results = {k: ci(v) for k, v in metrics.items()}
    results["point"] = {
        "acc": float((y == preds).mean()),
        "logloss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, probs)),
        "n": int(n),
    }

    print(f"  Accuracy: {results['point']['acc']:.3f} "
          f"95%CI [{results['acc']['ci_lo']:.3f}, {results['acc']['ci_hi']:.3f}]")
    return results


# ── 2. Worst-case analysis ───────────────────────────────────────────────────

def worst_case_analysis(features, model_pkg, test_seasons=range(2015, 2026)):
    print("\n[2/6] Worst-case analysis...")
    test_df = features[features["season"].isin(test_seasons)].copy()
    if len(test_df) == 0:
        return {}

    feat_cols = model_pkg["feature_cols"]
    avail = [c for c in feat_cols if c in test_df.columns]
    X = test_df[avail].fillna(0).reindex(columns=feat_cols, fill_value=0)
    y = test_df["label"].values
    probs = model_pkg["xgb_model"].predict_proba(X)[:, 1]
    test_df["correct"] = (y == (probs >= 0.5).astype(int))

    by_season = (test_df.groupby("season")["correct"]
                 .agg(["mean", "count"]).rename(columns={"mean": "acc", "count": "n"})
                 .sort_values("acc"))

    by_type = (test_df.groupby("game_type")["correct"]
               .agg(["mean", "count"]).rename(columns={"mean": "acc", "count": "n"})
               .sort_values("acc"))

    print("  Worst seasons:")
    for s, r in by_season.head(5).iterrows():
        print(f"    {s}: {r['acc']:.3f} ({int(r['n'])} games)")

    return {
        "worst_seasons": by_season.head(5).reset_index().to_dict("records"),
        "by_game_type": by_type.reset_index().to_dict("records"),
    }


# ── 3. Calibration audit ────────────────────────────────────────────────────

def calibration_audit(features, model_pkg, test_seasons=range(2015, 2026)):
    print("\n[3/6] Calibration audit...")
    test_df = features[features["season"].isin(test_seasons)].copy()
    if len(test_df) == 0:
        return {}

    feat_cols = model_pkg["feature_cols"]
    avail = [c for c in feat_cols if c in test_df.columns]
    X = test_df[avail].fillna(0).reindex(columns=feat_cols, fill_value=0)
    y = test_df["label"].values
    probs = model_pkg["xgb_model"].predict_proba(X)[:, 1]

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)

    reliability = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        reliability.append({
            "bin": float((bins[i] + bins[i + 1]) / 2),
            "predicted": float(probs[mask].mean()),
            "actual": float(y[mask].mean()),
            "n": int(mask.sum()),
            "error": float(abs(probs[mask].mean() - y[mask].mean())),
        })

    ece = sum(r["n"] / len(y) * r["error"] for r in reliability)
    mce = max(r["error"] for r in reliability) if reliability else 0

    print(f"  ECE: {ece:.4f}  MCE: {mce:.4f}")
    return {"reliability": reliability, "ece": float(ece), "mce": float(mce)}


# ── 4. Feature importance ───────────────────────────────────────────────────

def feature_importance(features, model_pkg, test_seasons=range(2015, 2026)):
    print("\n[4/6] Feature importance...")
    try:
        import shap
        test_df = features[features["season"].isin(test_seasons)].copy()
        feat_cols = model_pkg["feature_cols"]
        avail = [c for c in feat_cols if c in test_df.columns]
        X = (test_df[avail].fillna(0).reindex(columns=feat_cols, fill_value=0)
             .sample(min(500, len(test_df)), random_state=42))
        explainer = shap.TreeExplainer(model_pkg["xgb_model"])
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        imp = dict(zip(feat_cols, np.abs(shap_vals).mean(axis=0).tolist()))
        method = "shap"
    except ImportError:
        raw = model_pkg["xgb_model"].get_booster().get_fscore()
        total = sum(raw.values()) or 1
        imp = {k: v / total for k, v in raw.items()}
        method = "xgb_fscore"

    ranked = sorted(imp.items(), key=lambda x: -abs(x[1]))
    print("  Top 15:")
    for f, s in ranked[:15]:
        print(f"    {f:<35} {s:.4f}")

    return {"method": method, "importance": [{"f": f, "s": float(s)} for f, s in ranked]}


# ── 5. Predictor class ──────────────────────────────────────────────────────

class NBAGodPredictor:
    """Production prediction interface for NBA-God."""

    # ESPN sometimes uses abbreviated names — map them to canonical names
    ESPN_ALIASES = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
    }

    def __init__(self, model_pkg, team_names, features_df):
        self.xgb_model = model_pkg["xgb_model"]
        self.lr_model = model_pkg.get("lr_model")
        self.scaler = model_pkg.get("scaler")
        self.feat_cols = model_pkg["feature_cols"]
        self.ensemble_w = model_pkg.get("ensemble_weights", {"xgb": 0.65, "lr": 0.35})
        self.team_names = team_names
        self.name_to_id = {v.lower(): k for k, v in team_names.items()}
        self.features_df = features_df

    @classmethod
    def load(cls):
        pkg, features, team_names = load_artifacts()
        return cls(pkg, team_names, features)

    def _resolve_team(self, team):
        if isinstance(team, (int, np.integer)):
            return int(team)
        team_lower = str(team).lower()
        # Check ESPN aliases first
        team_lower = self.ESPN_ALIASES.get(team_lower, team_lower)
        if team_lower in self.name_to_id:
            return self.name_to_id[team_lower]
        matches = [(k, v) for k, v in self.name_to_id.items()
                   if team_lower in k or k in team_lower]
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous: '{team}' -> {[self.team_names[v] for _, v in matches[:5]]}")
        raise ValueError(f"Team not found: '{team}'")

    def predict(self, team_a, team_b, season: int = 2025) -> dict:
        tid_a = self._resolve_team(team_a)
        tid_b = self._resolve_team(team_b)
        name_a = self.team_names.get(tid_a, str(team_a))
        name_b = self.team_names.get(tid_b, str(team_b))

        feats = self.features_df[self.features_df["season"] == season]
        mask = (
            ((feats["team1_id"] == tid_a) & (feats["team2_id"] == tid_b)) |
            ((feats["team1_id"] == tid_b) & (feats["team2_id"] == tid_a))
        )

        if mask.any():
            row = feats[mask].iloc[-1]
            flipped = (row["team1_id"] == tid_b)
            avail = [c for c in self.feat_cols if c in row.index]
            X = pd.DataFrame([row[avail].fillna(0)]).reindex(columns=self.feat_cols, fill_value=0)
        else:
            X = pd.DataFrame([{c: 0 for c in self.feat_cols}])
            flipped = False

        # Ensemble prediction
        prob_xgb = float(self.xgb_model.predict_proba(X)[0, 1])
        if self.lr_model and self.scaler:
            X_scaled = self.scaler.transform(X)
            prob_lr = float(self.lr_model.predict_proba(X_scaled)[0, 1])
            prob = (self.ensemble_w["xgb"] * prob_xgb +
                    self.ensemble_w["lr"] * prob_lr)
        else:
            prob = prob_xgb

        if flipped:
            prob = 1 - prob

        confidence = "high" if prob >= 0.65 or prob <= 0.35 else (
            "medium" if prob >= 0.55 or prob <= 0.45 else "low"
        )

        return {
            "team_a": name_a, "team_b": name_b, "season": season,
            "prob_a_wins": round(prob, 4), "prob_b_wins": round(1 - prob, 4),
            "favored": name_a if prob >= 0.5 else name_b,
            "confidence": confidence,
        }

    def batch_predict(self, matchups, season=2025):
        return [self.predict(a, b, season) for a, b in matchups]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nNBA-God — Phase 5: Robustness & Deployment\n")

    model_pkg, features, team_names = load_artifacts()

    boot = bootstrap_ci(features, model_pkg)
    with open(OUT_DIR / "bootstrap_ci.json", "w") as f:
        json.dump(boot, f, indent=2)

    worst = worst_case_analysis(features, model_pkg)
    with open(OUT_DIR / "worst_case.json", "w") as f:
        json.dump(worst, f, indent=2, default=str)

    cal = calibration_audit(features, model_pkg)
    with open(OUT_DIR / "calibration_audit.json", "w") as f:
        json.dump(cal, f, indent=2)

    fi = feature_importance(features, model_pkg)
    with open(OUT_DIR / "feature_importance.json", "w") as f:
        json.dump(fi, f, indent=2)

    print("\n[5/6] Saving predictor...")
    predictor = NBAGodPredictor(model_pkg, team_names, features)
    with open(OUT_DIR / "predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)

    print("\n[6/6] Final report...")
    report = {
        "project": "NBA-God",
        "description": "NBA Game Prediction Engine (1946-2025)",
        "accuracy": boot.get("point", {}).get("acc"),
        "accuracy_95ci": boot.get("acc"),
        "calibration": {"ece": cal.get("ece"), "mce": cal.get("mce")},
        "top_features": [x["f"] for x in fi.get("importance", [])[:10]],
    }
    with open(OUT_DIR / "final_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("  PHASE 5 SUMMARY")
    print("=" * 60)
    if boot.get("point"):
        print(f"  Accuracy: {boot['point']['acc']:.3f}")
    if cal:
        print(f"  ECE: {cal.get('ece', 'N/A')}")
    print(f"  All outputs: {OUT_DIR}")
    print("=" * 60)
    print("\nNBA-God build complete. All 5 phases done.")


if __name__ == "__main__":
    main()
