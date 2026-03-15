# NBA-God Iteration Log

**Start Date:** March 14, 2026
**Goal:** Apply Basketball-God3.0 Phase 1 improvements; validate + optimize

---

## Baseline Assessment

From final_report.json:
- **Accuracy:** 92.78% (suspiciously high — suggests lookahead or overfitting)
- **Calibration ECE:** 0.041 (reasonable)
- **Top feature:** diff_h2h_win_pct (H2H dominates, expected for NBA)

---

## CRITICAL AUDIT: Is NBA-God Really 92.78% Accurate?

**Question:** Is this real or lookahead bias?

**Expected:**
- NBA is more predictable than college (established teams, consistency)
- 92% is possible but... let me verify walk-forward validation

**To Check:**
1. Does the model use future data (post-game scores, live odds)?
2. Are totals Over/Under bets losing money (like Basketball-God)?
3. Is injury feature adding noise?
4. What's ROI on actual betting signals?

---

## Phase 1 Applicability

### Fix #1: Disable Totals Bets
- **Status:** NBA-God doesn't generate totals signals in daily_predictor.py
- **Finding:** Missing opportunity or design choice?
- **Action:** Check if phase5_deploy produces totals predictions

### Fix #2: Disable Injury Feature
- **Status:** config.py references ROTOWIRE_INJURY_URL but impact unclear
- **Finding:** May not be weighted feature like Basketball-God3.0
- **Action:** Check phase2_features/build_features.py for injury columns

### Fix #3: CLV Tracker
- **Status:** Unknown if implemented
- **Action:** Check web/server.py for CLV logging

---

## Phase 1 Findings

### Fix #1: Disable Totals ❌
- **Status:** NOT APPLICABLE
- **Reason:** NBA-God doesn't generate totals betting signals
- **Evidence:** predictions_20260313.json contains only moneyline + spread

### Fix #2: Disable Injury Feature ❓
- **Status:** UNKNOWN if actively damaging
- **Action:** Will audit in Phase 2 (lower priority for NBA)

### Fix #3: CLV Tracker ✓
- **Status:** Likely needed; will check Phase 2

---

## NBA-God vs Basketball-God3.0 Structure

| Item | Basketball-God3.0 | NBA-God |
|------|------------------|---------|
| Totals bets | ❌ DISABLED in Phase 1 | ✓ Not generated |
| Injury feature | ❌ DISABLED in Phase 1 | ❓ May be present |
| Confidence | 60%/55% thresholds | High (94%+ common) |
| Accuracy | 70.2% → 70.4% v2 | 92.78% (needs validation) |

---

## Decision: Skip Direct Phase 1 Clone

NBA-God doesn't have the same problems:
- ✅ No totals bleeding money (doesn't generate them)
- ✓ Different feature set; injury less prominent
- ⚠️ 92.78% seems high; may have different issues

**Moving directly to Phase 2** which benefits both:
1. **Ensemble weight optimization** (both use 65/35 XGB/LR)
2. **Signal threshold calibration** (both use moneyline/spread)
3. **Train-serve skew audit** (both need validation)

---

## Phase 2 Priority for NBA-God

1. **Validate 92.78% accuracy** (lookahead bias check)
2. **Optimize ensemble weights** (likely 65/35 is not optimal for NBA)
3. **Calibrate thresholds** (94%+ confidence predictions need validation)
4. **Audit train-serve skew** (data pipelines may differ from basketball)

Expected gain: +3-8% ROI on top of current baseline
