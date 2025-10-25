/**
 * Quiz questions for Correlation & Association section
 */

export const correlationassociationQuiz = [
  {
    id: 'q1',
    question:
      'Two features in your dataset have a Pearson correlation of 0.92. One is "years_of_experience" and the other is "age". Should you keep both features in your linear regression model? Explain the consequences of keeping both vs. removing one.',
    hint: 'Think about multicollinearity and its effects on regression coefficients.',
    sampleAnswer:
      'With r=0.92, these features are highly correlated (multicollinearity). **Consequences of keeping both**: (1) **Unstable coefficients**: Small changes in data cause large coefficient swings. (2) **Inflated standard errors**: Wider confidence intervals, harder to detect significance. (3) **Incorrect interpretation**: Coefficients become meaningless - "holding age constant while varying experience" doesn\'t make sense when they\'re highly correlated. (4) **High VIF**: Variance Inflation Factor > 10, indicating severe multicollinearity. (5) **Model still predicts well**: Accuracy unaffected, but interpretation breaks down. **Solution options**: (1) **Remove one feature**: Keep more important one (domain knowledge). If removing experience, model might be: salary = β₀ + β₁(age) + ε. (2) **Combine features**: Create "career_stage" = weighted average. (3) **Use Ridge/Lasso**: Regularization handles multicollinearity by shrinking coefficients. Ridge especially good here. (4) **PCA**: Create orthogonal components from both. **When to keep both**: (1) Prediction-only models (Random Forest, XGBoost) - they handle correlation fine. (2) With regularization (Ridge/Lasso). (3) If interaction is meaningful. **Best practice for linear models**: Remove one feature or use regularization. The r=0.92 means they\'re nearly redundant - keeping both adds complexity without information gain.',
    keyPoints: [
      'r=0.92 indicates severe multicollinearity',
      'Causes unstable coefficients and inflated standard errors',
      'Prediction accuracy unaffected, interpretation breaks',
      'Solutions: remove one, combine, regularization, PCA',
      'Tree models handle multicollinearity better',
      'For linear models: keep one or use Ridge/Lasso',
    ],
  },
  {
    id: 'q2',
    question:
      'You find that ice cream sales and sunglasses sales have a correlation of 0.85. A colleague concludes "selling more ice cream causes people to buy sunglasses!" What\'s wrong with this conclusion? How would you explain the true relationship and what additional analysis would you perform?',
    hint: 'Think about confounding variables and establishing causation.',
    sampleAnswer:
      'This is a textbook case of "correlation ≠ causation"! **What\'s wrong**: Correlation only shows association, not causation. The colleague commits the "post hoc ergo propter hoc" fallacy (correlation implies causation). **True explanation**: Both are driven by a **confounding variable**: temperature/season! (1) Summer → hot weather → more ice cream sales. (2) Summer → sunny weather → more sunglasses sales. (3) Ice cream and sunglasses are correlated but don\'t cause each other. **Additional analyses**: (1) **Partial correlation**: Control for temperature. Compute corr (ice_cream, sunglasses | temperature). Likely drops to near zero! (2) **Time series analysis**: Show both peak in summer, trough in winter. (3) **Granger causality**: Does ice cream "predict" sunglasses? (Spoiler: No.) (4) **Experiment**: Run A/B test - promote ice cream in winter. Sunglasses sales won\'t increase. **Establishing causation requires**: (1) **Temporal precedence**: Cause must precede effect. (2) **Covariation**: They must be correlated (✓ we have this). (3) **No plausible alternatives**: Rule out confounders (temperature is the real cause). (4) **Experimental manipulation**: Randomized controlled trial. **In ML context**: This happens constantly! Model finds: (a) "Rain" correlates with "umbrella sales" AND "traffic accidents" - doesn\'t mean umbrellas cause accidents! (b) "High credit card usage" correlates with "default" - but both might be caused by "financial stress". Always ask: Is there a confounding variable? Could the relationship be spurious?',
    keyPoints: [
      'Correlation does not imply causation',
      'Both likely driven by confounding variable (temperature)',
      'Test with partial correlation controlling for confounder',
      'Causation requires: temporal precedence, covariation, no alternatives',
      'Experiments (RCTs) are gold standard for causation',
      'In ML: always consider confounders before interpreting',
    ],
  },
  {
    id: 'q3',
    question:
      'You compute Pearson correlation between two features and get r=0.15 (p=0.08). Your colleague says "the correlation is not significant (p>0.05) so there\'s no relationship." Is this correct? What factors might affect this conclusion?',
    hint: 'Consider effect size, power, sample size, and type of relationship.',
    sampleAnswer:
      'The colleague is wrong on multiple levels: **Incorrect interpretation of p>0.05**: (1) p=0.08 means "8% chance of seeing r=0.15 if true r=0". (2) Doesn\'t prove there\'s NO relationship - just insufficient evidence. (3) Absence of evidence ≠ evidence of absence! **Small effect size**: r=0.15 is weak but not zero. With large enough sample, might be meaningful. **Power and sample size**: (1) With small sample (say n=30): 80% power only detects |r|>0.48. The r=0.15 would be missed (Type II error). (2) With large sample (n=1000): Can detect even r=0.08 with high power. (3) Power analysis needed: "Did we have adequate sample to detect meaningful effects?" **Non-linear relationships**: (1) Pearson only detects LINEAR relationships. (2) Could be strong nonlinear relationship (quadratic, exponential). (3) Check: Spearman correlation, scatter plots. **Practical significance**: (1) Even if p<0.05, r=0.15 might be too weak to matter. (2) R² = 0.15² = 2.25% of variance explained - negligible! (3) Statistical significance ≠ practical importance. **Correct conclusion**: "With our sample size, we don\'t have sufficient evidence to conclude the correlation is non-zero. However, this doesn\'t prove there\'s no relationship. The observed r=0.15 is weak, and we\'d need n≈400 to have 80% power to detect it as significant. We should: (1) Collect more data, or (2) Accept that any relationship is too weak to be useful." **Best practice**: Always report: correlation, p-value, confidence interval, sample size, and power. Never conclude "no effect" from p>0.05 alone!',
    keyPoints: [
      'p>0.05 means insufficient evidence, NOT no relationship',
      'Small sample = low power to detect weak correlations',
      'r=0.15 is weak but not zero (explains 2.25% variance)',
      'Pearson only detects linear relationships',
      'Need power analysis: did we have adequate sample size?',
      'Statistical significance ≠ practical significance',
    ],
  },
];
