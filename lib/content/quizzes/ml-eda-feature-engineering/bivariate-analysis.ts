/**
 * Quiz questions for Bivariate Analysis section
 */

export const bivariateanalysisQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between Pearson, Spearman, and Kendall correlations. When would you use each, and what does it mean if Pearson and Spearman correlations differ significantly?',
    hint: 'Think about linearity vs monotonicity, and sensitivity to outliers.',
    sampleAnswer:
      'These three correlations measure different types of relationships: PEARSON CORRELATION (r): Measures LINEAR relationships between continuous variables. Range: [-1, 1]. Assumes: Both variables normally distributed, relationship is linear, no extreme outliers. WHEN TO USE: When you expect a straight-line relationship. Example: Height vs Weight. Sensitive to outliers and non-linearity. SPEARMAN CORRELATION (ρ): Measures MONOTONIC relationships using ranks instead of raw values. Range: [-1, 1]. Assumes: Monotonic relationship (consistently increasing or decreasing, but not necessarily linear). WHEN TO USE: Non-linear but monotonic data, ordinal data, presence of outliers. Example: Education level (ordinal) vs Income. Robust to outliers. KENDALL CORRELATION (τ): Also rank-based, measures ordinal association. Range: [-1, 1]. More robust than Spearman with small samples. WHEN TO USE: Small sample sizes, want robustness, ordinal data. INTERPRETATION WHEN THEY DIFFER: If |Pearson| >> |Spearman|: Impossible (Spearman often higher for monotonic). If |Spearman| >> |Pearson|: NON-LINEAR but MONOTONIC relationship. Example: Exponential growth (y=e^x). Pearson might be 0.6 but Spearman 0.95. If Pearson ≈ Spearman: LINEAR relationship. If Pearson = 0 but Spearman ≠ 0: Non-linear, non-monotonic relationship. Example: U-shaped curve (y=x²). PRACTICAL RULE: Always compute both. If they differ significantly, investigate with scatter plot - likely non-linear relationship that needs transformation (log, polynomial) or non-linear model.',
    keyPoints: [
      'Pearson: linear relationships, sensitive to outliers',
      'Spearman: monotonic relationships, rank-based, robust',
      'Kendall: also rank-based, better for small samples',
      'Large difference suggests non-linear but monotonic relationship',
      'Similar values suggest linear relationship',
      'Always visualize to understand the relationship type',
    ],
  },
  {
    id: 'q2',
    question:
      'You want to test if customer satisfaction scores differ significantly across three product categories. Walk through the complete statistical testing process, including assumption checks and choosing between parametric and non-parametric tests.',
    sampleAnswer:
      'Complete statistical testing process for comparing groups: STEP 1: VISUALIZE THE DATA First, create box plots and histograms for each category to visually inspect distributions and potential differences. STEP 2: CHECK ASSUMPTIONS FOR ANOVA (Parametric Test) ANOVA requires: (a) NORMALITY: Each group should be approximately normally distributed. Test: Shapiro-Wilk test for each group (if n < 5000). If p > 0.05 for all groups → likely normal. Visual: Q-Q plots. (b) HOMOGENEITY OF VARIANCE: Groups should have similar variances. Test: Levene\'s test. If p > 0.05 → variances are equal. (c) INDEPENDENCE: Samples should be independent (different customers in each category). STEP 3: CHOOSE APPROPRIATE TEST IF ASSUMPTIONS MET: Use ONE-WAY ANOVA. F-statistic tests if at least one group mean differs. If p < 0.05 → significant difference exists. Follow up with POST-HOC tests (Tukey HSD, Bonferroni) to identify which pairs differ. IF ASSUMPTIONS VIOLATED (non-normal or unequal variances): Use KRUSKAL-WALLIS TEST (non-parametric alternative). H-statistic tests if distributions differ. If p < 0.05 → significant difference. Follow up with Mann-Whitney U tests (with Bonferroni correction) for pairwise comparisons. STEP 4: CALCULATE EFFECT SIZE Statistical significance (p-value) tells if effect is real, but not how large. Eta-squared (η²) for ANOVA: proportion of variance explained by groups. Small: 0.01, Medium: 0.06, Large: 0.14. Epsilon-squared (ε²) for Kruskal-Wallis. STEP 5: INTERPRET RESULTS Example: "One-way ANOVA revealed a statistically significant difference in satisfaction scores across product categories (F(2, 297) = 15.4, p < 0.001, η² = 0.09). Post-hoc Tukey HSD tests showed Category A (M=7.8, SD=1.2) scored significantly higher than Category B (M=6.5, SD=1.5, p < 0.001) and Category C (M=6.9, SD=1.3, p = 0.02), but Categories B and C did not differ significantly (p = 0.45)." STEP 6: BUSINESS INTERPRETATION Don\'t just report statistics! What does this mean for the business? "Category A products have substantially higher customer satisfaction. Investigate what Category A does differently and replicate across other categories."',
    keyPoints: [
      'Start with visualization (box plots, histograms)',
      'Check normality (Shapiro-Wilk) and variance homogeneity (Levene)',
      'Use ANOVA if assumptions met, Kruskal-Wallis if violated',
      'Calculate effect size (eta-squared) not just p-value',
      'Perform post-hoc tests to identify which groups differ',
      'Interpret results in business context, not just statistically',
    ],
  },
  {
    id: 'q3',
    question:
      'A colleague says "Feature A and Feature B have a correlation of 0.85, so Feature B causes Feature A to increase." Explain what\'s wrong with this statement and discuss the limitations of correlation analysis.',
    sampleAnswer:
      "This statement confuses CORRELATION with CAUSATION - one of the most common and dangerous errors in data science. What's wrong: (1) CORRELATION DOES NOT IMPLY CAUSATION: Correlation of 0.85 means A and B move together, but doesn't tell us if A causes B, B causes A, both are caused by C (confounding), or it's pure coincidence. Classic example: Ice cream sales correlate with drowning deaths (both caused by hot weather, not each other). (2) DIRECTION OF CAUSALITY UNKNOWN: Even if causal relationship exists, correlation doesn't tell direction. High income correlates with home ownership - but does high income cause home buying, or does home ownership lead to wealth accumulation (home equity)? Could be both. (3) CONFOUNDING VARIABLES: A third variable might cause both. Example: Shoe size correlates with reading ability in children (both caused by age). LIMITATIONS OF CORRELATION ANALYSIS: (1) ONLY LINEAR RELATIONSHIPS (Pearson): Can be zero even with strong non-linear relationship. Y = X² has Pearson correlation ≈ 0 but perfect functional relationship. (2) OUTLIERS INFLUENCE HEAVILY: Single extreme point can create/destroy correlation. (3) RANGE RESTRICTION: If you only look at narrow range, correlation appears weaker. Example: Height vs weight correlation lower if you only study adults vs all ages. (4) SIMPSON'S PARADOX: Correlation can reverse when data is aggregated. Trend within groups differs from overall trend. (5) SPURIOUS CORRELATIONS: Totally unrelated variables can correlate by chance, especially with small samples or data mining many variables. TO ESTABLISH CAUSATION, NEED: (1) Randomized Controlled Experiments (gold standard). (2) Temporal precedence (cause before effect). (3) Eliminate confounders (statistical controls, matching). (4) Mechanistic understanding (why would A cause B?). (5) Dose-response relationship. (6) Consistency across studies. IN ML CONTEXT: Correlation is useful for prediction (don't need causation), but for business decision-making (interventions), must understand causality. \"If we change A, will B change?\" requires causal thinking, not just correlation.",
    keyPoints: [
      'Correlation ≠ causation (fundamental principle)',
      'Cannot determine direction of causality from correlation',
      'Confounding variables may cause both correlated variables',
      'Correlation only captures linear relationships (Pearson)',
      "Outliers, range restriction, and Simpson's paradox affect correlation",
      'Prediction needs correlation; intervention needs causation',
    ],
  },
];
