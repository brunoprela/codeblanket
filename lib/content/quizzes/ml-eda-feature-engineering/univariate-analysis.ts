/**
 * Quiz questions for Univariate Analysis section
 */

export const univariateanalysisQuiz = [
  {
    id: 'q1',
    question:
      'Explain the relationship between skewness, mean, and median. How can you use the relationship between mean and median to quickly assess the shape of a distribution?',
    hint: 'Think about how outliers in different directions affect mean vs median.',
    sampleAnswer:
      "The relationship between mean and median reveals distribution skewness because they respond differently to extreme values: The median is robust to outliers (it's just the middle value), while the mean is sensitive to extremes (it's an average of all values). In a SYMMETRIC distribution: mean ≈ median (both around center). In a RIGHT-SKEWED distribution: mean > median (extreme high values pull the mean right, but median stays near center of mass). Example: household income where billionaires pull mean up dramatically. In a LEFT-SKEWED distribution: mean < median (extreme low values pull mean left). Example: test scores where most score high but a few fail. Quick assessment rule: If mean > median by more than ~5% of the median → right-skewed. If median > mean by more than ~5% of the median → left-skewed. If they're close → approximately symmetric. This is incredibly useful in EDA: (1) You can spot skewness instantly without computing skewness statistic. (2) Helps decide if log/sqrt transformation is needed (right-skew). (3) Informs which measure of central tendency to report (use median for skewed data as it's more representative). (4) Suggests which outlier direction to investigate. Real example: If median house price is $300K but mean is $450K, you know expensive outliers exist, and median better represents \"typical\" house.",
    keyPoints: [
      'Mean is sensitive to outliers, median is robust',
      'Right-skewed: mean > median (high outliers pull mean)',
      'Left-skewed: mean < median (low outliers pull mean)',
      'Symmetric: mean ≈ median',
      'Quick skewness assessment without formal statistics',
      'Guides transformation decisions and outlier investigation',
    ],
  },
  {
    id: 'q2',
    question:
      "You've detected outliers using four different methods (IQR, Z-score, Modified Z-score, Percentile) and they give different counts. Which method should you use and why? How do you decide whether to remove, transform, or keep outliers?",
    sampleAnswer:
      'Different outlier detection methods serve different purposes and no single method is always best. Choice depends on distribution and context: (1) IQR METHOD (1.5×IQR rule): Best for skewed data and general purpose. Assumes nothing about distribution. Typically identifies ~0.7% of normal data as outliers. USE WHEN: Distribution is non-normal or unknown. (2) Z-SCORE METHOD (|z| > 3): Best for normally distributed data. Assumes normal distribution. Flags ~0.3% of normal data. USE WHEN: Data is approximately normal. (3) MODIFIED Z-SCORE (using MAD): Best when outliers affect mean/std. Uses median instead of mean (robust). USE WHEN: Extreme outliers would distort Z-score calculation. (4) PERCENTILE METHOD: Arbitrary cutoff (e.g., 1st/99th percentile). Flags exactly the percentage you specify. USE WHEN: You need specific outlier percentage. DECISION FRAMEWORK FOR HANDLING OUTLIERS: (1) INVESTIGATE FIRST: Are they data errors? → Remove. Valid but extreme? → Keep or transform. Impossible values (age=200)? → Remove. (2) DOMAIN KNOWLEDGE: Does $10M house make sense in Beverly Hills? Yes → Keep. Does $10M house in rural Iowa make sense? No → Investigate. (3) IMPACT ANALYSIS: Does including/excluding change insights? If yes → investigate further. (4) TRANSFORMATION: Right-skewed with high outliers → log transform often brings them in. (5) MODEL CHOICE: Tree-based models (Random Forest, XGBoost) → robust to outliers → keep. Linear regression → sensitive → remove or transform. GOLDEN RULE: Never automatically remove outliers without investigation. Sometimes outliers are your most interesting data points (fraud detection, rare disease).',
    keyPoints: [
      'IQR method: best for skewed data, distribution-free',
      'Z-score: best for normal data, assumes normality',
      'Modified Z-score: robust when outliers affect mean/std',
      'Must investigate before removing - could be errors or valid extremes',
      'Consider transformation (log) instead of removal',
      'Model choice matters: trees are robust, linear models sensitive',
    ],
  },
  {
    id: 'q3',
    question:
      'Why do we test for normality, and what are the implications if our data is not normally distributed? What transformations would you apply for different types of non-normal distributions?',
    hint: 'Think about which ML algorithms and statistical tests assume normality.',
    sampleAnswer:
      "Normality testing is crucial because many ML algorithms and statistical methods assume normal distributions: WHY TEST FOR NORMALITY: (1) Many statistical tests (t-test, ANOVA, F-test) assume normality - violations lead to invalid p-values. (2) Linear regression assumes residuals are normal - violation affects confidence intervals and hypothesis tests. (3) Some ML algorithms perform better with normalized features. (4) Understanding distribution shape guides feature engineering. NORMALITY TESTS: (1) Visual: Q-Q plot (points should follow line), histogram (should be bell-shaped). (2) Statistical: Shapiro-Wilk (best for n<5000), Kolmogorov-Smirnov, Anderson-Darling. If p-value < 0.05 → reject normality. IMPLICATIONS OF NON-NORMALITY: (1) Linear models: predictions still work but confidence intervals unreliable. (2) Statistical tests: may give incorrect p-values (Type I/II errors). (3) Outlier sensitivity: non-normal data often has heavy tails (more outliers). (4) Feature scaling: standardization assumes normality for optimal results. TRANSFORMATIONS BY DISTRIBUTION TYPE: (1) RIGHT-SKEWED (long right tail, mean > median): → Log transformation: log(x) or log(x+1). → Square root: √x. → Inverse: 1/x. Examples: income, property prices, count data. (2) LEFT-SKEWED (long left tail, mean < median): → Square: x². → Exponential: e^x. Less common in practice. (3) HEAVY TAILS (high kurtosis): → Winsorization: cap extreme values at percentiles. → Rank transformation: convert to ranks. (4) GENERAL PURPOSE: → Box-Cox: automatically finds optimal power transformation (λ). → Yeo-Johnson: like Box-Cox but handles negative values. WHEN NORMALITY DOESN'T MATTER: Tree-based models (Random Forest, XGBoost, LightGBM) don't assume normality → no transformation needed. Neural networks with enough capacity → less critical. Non-parametric tests (Mann-Whitney, Kruskal-Wallis) → distribution-free.",
    keyPoints: [
      'Many statistical tests and linear models assume normality',
      'Non-normality affects confidence intervals and p-values',
      'Right-skewed → log/sqrt transformation',
      'Left-skewed → square/exponential transformation',
      'Box-Cox and Yeo-Johnson auto-find optimal transformation',
      "Tree-based models don't require normality",
    ],
  },
];
