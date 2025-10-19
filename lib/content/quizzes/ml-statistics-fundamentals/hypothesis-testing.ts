/**
 * Quiz questions for Hypothesis Testing section
 */

export const hypothesistestingQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between Type I and Type II errors in the context of deploying a machine learning model to production. Which error is typically more costly in high-stakes applications like medical diagnosis or fraud detection?',
    hint: 'Think about false positives vs false negatives in production deployments.',
    sampleAnswer:
      "Type I and Type II errors have different costs in ML deployment: **Type I Error (False Positive)**: Rejecting H₀ when it's true = Deploying a model that's actually no better than baseline. Consequences: (1) Wasted engineering effort migrating to new model. (2) Potential performance degradation if new model is actually worse. (3) Opportunity cost of not pursuing better alternatives. (4) Reduced trust if model fails after deployment. **Type II Error (False Negative)**: Failing to reject H₀ when it's false = NOT deploying a model that's actually better. Consequences: (1) Missing performance improvements and business value. (2) Competitors may deploy better models. (3) Continued suboptimal predictions. **Cost depends on domain**: (1) **Medical diagnosis**: Type I error often worse - deploying an unvalidated model could harm patients. Better to be conservative (require strong evidence p<<0.05) even if it means slower adoption of improvements. (2) **Fraud detection**: Type II error can be very costly - failing to deploy a better fraud detector means millions in losses. Here, you might accept p<0.10 to catch improvements faster. (3) **Recommendation systems**: Type I error relatively low cost - easy to roll back, mistakes not life-threatening. Can be more aggressive in testing new models. **Best practice**: Use power analysis to minimize Type II errors while controlling Type I. For high-stakes: require p<0.01 and larger sample sizes to be confident.",
    keyPoints: [
      "Type I: Deploy model that's not actually better (false positive)",
      "Type II: Don't deploy model that is actually better (false negative)",
      'Medical/safety: Type I error more costly (conservative threshold)',
      'Business/revenue: Type II error more costly (miss improvements)',
      'Use power analysis to balance both error types',
      'Adjust α based on deployment risk and cost',
    ],
  },
  {
    id: 'q2',
    question:
      'You conduct 20 hypothesis tests to evaluate 20 different features for inclusion in your model. One feature shows p=0.03, leading you to reject H₀ and include it. Why is this problematic, and how should you correct for multiple testing?',
    hint: 'Think about the family-wise error rate and multiple comparison problem.',
    sampleAnswer:
      "This is the **multiple testing problem** (p-hacking). The issue: With α=0.05, each test has a 5% false positive rate. With 20 tests, even if ALL features are useless (all H₀ true), expected false positives = 20 × 0.05 = 1. So finding one p<0.05 is exactly what we'd expect by chance alone! The feature might have no real predictive value. **Why problematic**: (1) Inflated Type I error rate. (2) False discoveries lead to overfitting. (3) Model won't generalize to new data. (4) Wasted computation on spurious features. **Corrections**: (1) **Bonferroni correction**: Divide α by number of tests. Use α' = 0.05/20 = 0.0025. Only include feature if p<0.0025. Very conservative, controls family-wise error rate. (2) **False Discovery Rate (FDR) / Benjamini-Hochberg**: Less conservative, controls proportion of false discoveries. Better for exploratory analysis. (3) **Hold-out validation**: Use separate validation set to confirm feature importance. (4) **Regularization**: Use L1/L2 instead of hypothesis tests for feature selection. **Best approach for ML**: Don't rely solely on p-values for feature selection. Use: (a) Cross-validation with hold-out test set. (b) Regularization (Lasso automatically selects features). (c) Proper train/val/test split - test set untouched until final evaluation. (d) If using p-values, apply FDR correction and confirm on validation set.",
    keyPoints: [
      'Multiple testing inflates false positive rate',
      'Expected false positives = n_tests × α',
      'Bonferroni: divide α by number of tests (conservative)',
      'FDR: controls proportion of false discoveries (less conservative)',
      'Better: use cross-validation and regularization',
      'Reserve test set for final evaluation only',
    ],
  },
  {
    id: 'q3',
    question:
      'Your A/B test comparing Model A and Model B shows p=0.04, just barely significant. Sample size was 100 for each model. Should you immediately deploy Model B? What other factors should you consider beyond statistical significance?',
    hint: 'Consider effect size, practical significance, power, and replication.',
    sampleAnswer:
      "P=0.04 is statistically significant at α=0.05, but DON'T rush to deploy! Other critical factors: **1. Effect Size**: How much better is Model B? If improvement is 75.0% → 75.1% (0.1%), it's statistically significant but practically worthless. Not worth the deployment cost. Look for: (a) Absolute improvement (percentage points). (b) Relative improvement (% increase). (c) Business impact (revenue, cost savings). **2. Sample Size & Power**: n=100 per group is relatively small. With small samples: (a) Estimates are noisy (wide confidence intervals). (b) Low statistical power - might miss true effect or find false positives. (c) P=0.04 is marginal (close to threshold). Calculate: (a) Confidence interval for the difference. (b) Power analysis - did you have adequate power (>80%) to detect meaningful effects? **3. Practical Significance**: (a) Is the improvement worth deployment cost? (b) Migration effort, monitoring, potential bugs. (c) Compare to minimum detectable effect you care about. **4. Replication**: (a) Run test for longer to increase sample size. (b) Confirm effect persists over time. (c) Test on different user segments. **5. Other Metrics**: (a) Check latency, cost, fairness. (b) Model B might be more accurate but 10x slower. **Recommendation**: (1) Continue A/B test to n=500+ per group. (2) Calculate confidence interval for improvement. (3) If lower bound of CI exceeds practical threshold, deploy. (4) Implement gradual rollout (10%→50%→100%). (5) Monitor closely for degradation. **Example decision rule**: Deploy if: p<0.05 AND improvement>2% AND n>500 AND 95% CI lower bound > 1%.",
    keyPoints: [
      'Statistical significance ≠ practical significance',
      'Small sample (n=100) gives noisy estimates',
      'Check effect size and confidence intervals',
      'Consider deployment cost vs benefit',
      'Increase sample size for more confident decision',
      'Gradual rollout with monitoring',
    ],
  },
];
