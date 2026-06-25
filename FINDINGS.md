# Findings — Insurance Claim-Settlement Bias Analysis

**Dataset:** 1,790 death claims · **Target:** `POLICY_STATUS` (Approved Death Claim vs Repudiate Death)
**Overall approval rate:** 68.0%

---

## 1. Descriptive cross-tabulation (vs policy status)

| Dimension | Highest approval | Lowest approval |
|---|---|---|
| Payment mode | Single premium ≈ 90% | Quarterly ≈ 45% |
| Underwriting | Medical ≈ 81% | Non-medical ≈ 66% |
| Claim timing | EARLY ≈ 77% | NON-EARLY ≈ 63% |
| Gender | Female ≈ 71% | Male ≈ 67% |

## 2. Diagnostic analysis — bias signals

**a) Channel / "team" disparity — strongest signal.**
TEAM-branded channels approve **85.4%** of claims vs **64.0%** for everyone else,
a **21.4-point** gap. Several TEAM zones (HIMALAYAN, AMRITSAR, DEHRADUN) exceed 90%,
while JKB JAMMU (~44%) and South (~47%) sit far below the 68% line. A Chi-square test
of `IS_TEAM_ZONE` vs outcome is highly significant (p ≪ 0.05), so the gap is not random.

**b) Income gradient.** The 1–100k income band is approved at only ~48%, below every
higher band — a possible documentation-burden effect on lower-income claimants.

**c) Early-claim counter-intuition.** EARLY claims (death soon after inception) are
approved *more* often than NON-EARLY claims — the reverse of the usual fraud-risk
expectation, and a process anomaly worth probing.

**d) Age.** Approval is broadly flat (63–69%) across 30–70, with the youngest (<30)
and oldest (70+) bands somewhat higher; age is a weaker signal than channel or income.

## 3. Feature engineering & models

Engineered: numeric parsing of money fields, `LOG_SUM_ASSURED`, `COVER_TO_INCOME`,
`INCOME_DECLARED`, `HAS_CLAIM_REASON`, `IS_TEAM_ZONE`; zone casing merged; rare
categories grouped; one-hot encoding; standardisation for KNN. Final matrix:
1,790 × 85 features, 75/25 stratified split.

## 4. Model performance (test set)

| Model | Train acc | Test acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| KNN | 0.744 | 0.699 | 0.735 | 0.872 | 0.798 | 0.700 |
| Decision Tree | 0.788 | 0.732 | 0.785 | 0.836 | 0.810 | 0.751 |
| Random Forest | 0.826 | 0.746 | 0.764 | 0.905 | 0.829 | 0.784 |
| **Gradient Boosting** | 0.868 | **0.768** | 0.793 | 0.892 | **0.840** | **0.788** |

Gradient Boosting is the most stable/best performer; KNN trails. The modest
train-vs-test gap on the ensembles indicates limited overfitting at these settings.
Top predictive drivers: sum assured, occupation, age, cover-to-income, and
`IS_TEAM_ZONE` — confirming the descriptive disparities are systematic.

## 5. Recommendations

1. **Audit the channel gap first.** Pull a stratified sample of repudiated claims from
   low-approval zones and approved claims from high-approval TEAM zones; review against
   identical underwriting criteria.
2. **Check low-income repudiations** for documentation/process friction.
3. **Investigate the EARLY-claim anomaly** — why are early claims approved more readily?
4. **Standardise decisioning** with a documented rubric and periodic disparity monitoring.

> **Caveat:** statistical disparity is not, by itself, proof of unfair bias. Differences
> may partly reflect legitimate factors (medical evidence, documentation quality, policy
> vintage). These findings are **prioritised audit leads**, not conclusions of wrongdoing.
