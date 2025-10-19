/**
 * Quiz questions for Common Statistical Tests section
 */

export const commonstatisticaltestsQuiz = [
  {
    id: 'q1',
    question:
      'You want to compare the accuracy of your model before and after hyperparameter tuning, using the same 10-fold cross-validation splits. Should you use a paired t-test or a two-sample t-test? Explain why this choice matters and what could go wrong with the other approach.',
    hint: 'Think about whether the measurements are independent or related.',
    sampleAnswer:
      "You should use a **paired t-test** because the measurements are related - they're from the same CV folds. Why this matters: **Paired t-test is correct**: (1) Each fold provides a matched pair: (before_fold_1, after_fold_1), ..., (before_fold_10, after_fold_10). (2) Reduces variance by focusing on within-fold differences. (3) Controls for fold difficulty - some folds are naturally harder. (4) More statistical power to detect improvements. (5) Tests if improvement is consistent across folds. **Two-sample t-test would be wrong**: (1) Treats 20 independent measurements (10 before + 10 after). (2) Ignores the paired nature - throws away information. (3) Higher variance in estimates. (4) Less power - harder to detect true improvements. (5) Violates independence assumption (folds from same dataset). **Example impact**: Suppose actual improvement is small but consistent (0.02 on every fold). Paired test will detect this because it looks at differences within folds. Two-sample test might miss it due to high between-fold variance. **Implementation**: Use stats.ttest_rel(after, before) NOT stats.ttest_ind(). **General principle**: Use paired tests whenever measurements come from the same subjects/units/folds. This is common in ML when using CV, comparing models on same test set, or evaluating before/after interventions.",
    keyPoints: [
      'Use paired t-test for same CV folds (related measurements)',
      'Paired test controls for fold-specific difficulty',
      'More statistical power by reducing variance',
      'Two-sample test ignores pairing, loses information',
      'Paired test checks consistency of improvement',
      'Common in CV, same test set, before/after comparisons',
    ],
  },
  {
    id: 'q2',
    question:
      "You're comparing three different models (Random Forest, XGBoost, Neural Net) using ANOVA and get p=0.03. Can you conclude which models are different? What additional tests are needed, and why must you adjust for multiple comparisons?",
    hint: "ANOVA tells you IF there's a difference, not WHERE the difference is.",
    sampleAnswer:
      "ANOVA with p=0.03 tells you that **at least one model is different**, but NOT which ones! You need post-hoc tests: **What ANOVA tells you**: Reject H₀ that all means are equal. At least one model has significantly different performance. But could be: (a) RF ≠ XGB ≠ NN (all different), (b) RF ≠ XGB = NN (RF different), (c) RF = XGB ≠ NN (NN different). **Post-hoc tests needed**: Pairwise comparisons: (1) RF vs XGB, (2) RF vs NN, (3) XGB vs NN. This means 3 t-tests. **Why adjust for multiple comparisons**: With α=0.05 per test: P(at least one false positive) = 1 - (0.95)³ ≈ 14.3%. We're doing 3 tests, so false positive rate inflates! **Correction methods**: (1) **Bonferroni** (conservative): α' = 0.05/3 = 0.0167 per test. Simple but potentially too strict. (2) **Tukey HSD** (balanced): Controls family-wise error rate, less conservative than Bonferroni. (3) **Holm-Bonferroni** (sequential): More powerful than Bonferroni. (4) **FDR control**: If you're okay with some false positives (exploratory). **Implementation**: from scipy.stats import tukey_hsd; result = tukey_hsd(model_a, model_b, model_c). **Best practice**: (1) Run ANOVA first (omnibus test). (2) Only if significant, do pairwise comparisons. (3) Apply multiple testing correction. (4) Report both raw and adjusted p-values. Without correction, you're p-hacking and will find spurious differences!",
    keyPoints: [
      'ANOVA tests if ANY difference exists, not which pairs differ',
      'Need post-hoc pairwise tests after significant ANOVA',
      'Multiple comparisons inflate Type I error rate',
      'Bonferroni: divide α by number of tests',
      'Tukey HSD: balanced correction for ANOVA',
      'Always report correction method used',
    ],
  },
  {
    id: 'q3',
    question:
      'Your data is highly skewed (accuracy distributions are exponential-like, not normal). Should you use a t-test or a non-parametric test? Under what conditions might a t-test still be appropriate despite non-normality?',
    hint: 'Consider sample size and the Central Limit Theorem.',
    sampleAnswer:
      "This depends on sample size, thanks to the Central Limit Theorem: **Non-parametric test (Mann-Whitney, Wilcoxon)** is safer when: (1) **Small sample** (n < 30 per group): CLT doesn't apply yet, t-test assumes normality. (2) **Extreme skewness/outliers**: Even with larger n, outliers distort means. (3) **Ordinal data**: Ranks, Likert scales. (4) **Unknown distribution**: Better to be conservative. **T-test might still be okay** despite non-normality when: (1) **Large sample** (n > 30, ideally n > 50): CLT kicks in - sampling distribution of means becomes normal even if data isn't. (2) **Mild to moderate skewness**: t-test is robust to modest departures from normality. (3) **Symmetric but not normal**: t-test handles deviations better than extreme skew. (4) **Equal sample sizes**: t-test more robust when groups have equal n. **How to decide**: (1) Check normality: Shapiro-Wilk test, Q-Q plot. (2) If p_shapiro < 0.05 AND n < 30: Use non-parametric. (3) If p_shapiro < 0.05 BUT n > 50: t-test probably okay, but check. (4) If severe outliers: Use non-parametric or robust methods. **Trade-offs**: Non-parametric tests: (a) Fewer assumptions, more robust. (b) Test medians/distributions, not means. (c) Less power if data actually normal. T-tests: (a) More power if assumptions met. (b) Tests means (often more interesting). (c) Better developed confidence intervals. **Best practice**: Run both! If they agree (both significant or both not), conclusion is robust. If they disagree, investigate why (outliers, skewness) and prefer non-parametric for safety.",
    keyPoints: [
      'Small sample (n<30) + non-normal → use non-parametric',
      'Large sample (n>50) → t-test okay even if non-normal (CLT)',
      'Check assumptions: Shapiro-Wilk test, Q-Q plots',
      'Non-parametric: fewer assumptions, tests medians',
      'T-test: more power if assumptions met, tests means',
      'Run both tests - if they agree, conclusions robust',
    ],
  },
];
