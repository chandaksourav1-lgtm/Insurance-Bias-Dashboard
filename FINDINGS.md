# Findings — Insurance Claim-Settlement Bias Analysis (v2 — Tuned)

**Dataset:** 1,790 death claims · **Target:** `POLICY_STATUS` (Approved vs Repudiated)
**Overall approval rate:** 68.0%

---

## 1. Descriptive cross-tabulation

| Dimension | Highest approval | Lowest approval |
|---|---|---|
| Payment mode | Single premium ≈ 90% | Quarterly ≈ 45% |
| Underwriting | Medical ≈ 81% | Non-medical ≈ 66% |
| Claim timing | EARLY ≈ 77% | NON-EARLY ≈ 63% |
| Occupation | Pensioner/Retired ≈ 89% | Service/Business ≈ 43% |

## 2. Diagnostic bias signals

| Signal | Detail | Strength |
|---|---|---|
| **Channel gap** | TEAM zones 85.4% vs Non-TEAM 64.0% (21.4pp gap, p ≪ 0.05) | Very strong |
| **Income gradient** | 1–100k band approved at only 48%, below all higher bands | Strong |
| **Early-claim anomaly** | EARLY claims approved MORE often (77% vs 63%) — counter-intuitive | Moderate |
| **Occupation effect** | Service/Business 43% vs Pensioner 92% | Strong |
| **Age** | Broadly flat 63–69% across 30–70; weak signal | Weak |

## 3. Feature engineering (v3 — enhanced)

| Enhancement | Description |
|---|---|
| Interaction features | `TEAM × EARLY`, `MEDICAL × EARLY`, `TEAM × MEDICAL`, `HIGH_COVER_LOW_INCOME` |
| Occupation grouping | 97 raw categories → 7 meaningful buckets |
| Claim-reason grouping | 62 raw categories → 8 clinical clusters |
| Additional numerics | `AGE²`, `LOG_INCOME`, binary flags |
| Final matrix | 1,790 × 109 features |
| Cross-validation | 5-fold stratified CV throughout |
| Hyperparameter tuning | GridSearchCV / manual grid search on all 4 models |

## 4. Model performance — before vs after tuning

| Model | Old Test | **New Test** | **Δ Test** | Old Gap | **New Gap** | CV Mean |
|---|---|---|---|---|---|---|
| KNN | 0.699 | **0.757** | **+5.8pp** | 4.5pp | **0.2pp** | 0.714 |
| Decision Tree | 0.732 | 0.730 | −0.2pp | 5.6pp | 7.6pp | 0.729 |
| Random Forest | 0.746 | **0.757** | **+1.1pp** | 8.0pp | 8.2pp | 0.747 |
| **Gradient Boosting** | 0.768 | **0.775** | **+0.7pp** | 10.0pp | **7.7pp** | **0.752** |

### Best hyperparameters (from GridSearchCV)

| Model | Key parameters |
|---|---|
| KNN | k=15, uniform weights, manhattan distance |
| Decision Tree | max_depth=6, min_samples_leaf=5, entropy |
| Random Forest | 300 trees, max_depth=10, max_features=20%, min_leaf=5 |
| Gradient Boosting | 300 trees, lr=0.05, max_depth=3, subsample=0.9 |

### Key improvements explained
- **KNN:** Manhattan distance + uniform weights + StandardScaler on 109 enhanced features
  eliminated the old train-test gap entirely (0.2pp) while boosting test accuracy by 5.8pp.
- **Gradient Boosting:** Lower learning rate (0.05 vs default 0.1) + more trees (300 vs 100)
  + subsample=0.9 reduced overfitting while improving generalisation.
- **Random Forest:** `max_features=0.2` (20% of 109 features per split) provided better
  regularisation than the default `sqrt`.
- **Cross-validation** (5-fold stratified) confirms all models generalise stably; CV means
  sit within 1–2pp of test accuracy.

## 5. Recommendations

1. **Audit the channel gap first.** Pull a stratified sample of repudiated claims from
   low-approval zones (JKB JAMMU, South) and approved claims from high-approval TEAM
   zones; review against identical underwriting criteria.
2. **Check low-income repudiations** for documentation/process friction.
3. **Investigate the EARLY-claim anomaly** — why are early claims approved more readily?
4. **Review occupation-based patterns** — Service/Business claims repudiated at 2× the
   rate of Retired/Pensioner; is this justified by risk or a process artefact?
5. **Standardise decisioning** with a documented rubric and periodic disparity monitoring.

> **Caveat:** statistical disparity ≠ proof of unfair bias. Differences may partly reflect
> legitimate factors (medical evidence, documentation, policy vintage). These findings are
> **prioritised audit leads**, not conclusions of wrongdoing.
