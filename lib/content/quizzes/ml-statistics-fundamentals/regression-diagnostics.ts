/**
 * Quiz questions for Regression Diagnostics section
 */

export const regressiondiagnosticsQuiz = [
  {
    id: 'q1',
    question:
      'Your residual vs fitted plot shows a clear funnel shape (variance increasing). How does this affect your model, and what should you do?',
    hint: 'Think about homoscedasticity assumption and its consequences.',
    sampleAnswer:
      'Funnel shape indicates **heteroscedasticity** - non-constant variance. **Effects**: (1) Coefficient estimates remain unbiased but inefficient. (2) Standard errors are wrong → confidence intervals and p-values invalid. (3) May over/under-estimate uncertainty. **Remedies**: (1) Transform Y (log, sqrt), (2) Weighted Least Squares, (3) Robust standard errors. For ML: predictions still work but intervals unreliable.',
    keyPoints: [
      'Heteroscedasticity detected',
      'Standard errors invalid',
      'Transform Y or use WLS',
      'Predictions unaffected',
      'Use robust standard errors',
    ],
  },
  {
    id: 'q2',
    question:
      "One observation has Cook's Distance of 2.5 (threshold is 0.08). Should you remove it? What factors should you consider?",
    hint: 'Consider data quality, domain knowledge, and sensitivity analysis.',
    sampleAnswer:
      "Cook's D = 2.5 >> 0.08 indicates highly influential point. **Before removing**: (1) Investigate: data error? True outlier? (2) Check leverage and residual. (3) Run model with/without - how much changes? (4) Domain knowledge: is this value plausible? **Decision**: Remove if data error confirmed. Keep if legitimate rare case but report sensitivity. Use robust regression if uncertain. Always document decisions and show both analyses.",
    keyPoints: [
      'Investigate before removing',
      'Check if data error',
      'Sensitivity analysis',
      'Domain knowledge crucial',
      'Document decisions',
    ],
  },
  {
    id: 'q3',
    question:
      'Q-Q plot tails deviate but n=500. Should you worry about non-normality for inference?',
    hint: 'Consider Central Limit Theorem and sample size.',
    sampleAnswer:
      'With n=500, **less concerning** due to CLT. **Analysis**: (1) Large n makes inference robust to modest non-normality. (2) Tail deviations suggest heavy tails or outliers. (3) Point predictions unaffected. (4) Confidence intervals may be slightly off. **Recommendation**: Check for outliers, consider robust standard errors, but likely okay. For small n (<30), non-normality is more problematic. Always report diagnostics.',
    keyPoints: [
      'Large n → CLT applies',
      'Inference robust to modest non-normality',
      'Check for outliers',
      'Predictions unaffected',
      'Use robust SEs for safety',
    ],
  },
];
