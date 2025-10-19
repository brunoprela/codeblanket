/**
 * Quiz questions for Experimental Design section
 */

export const experimentaldesignQuiz = [
  {
    id: 'q1',
    question:
      'Why is randomization critical in A/B testing? What would happen without it?',
    hint: 'Think about confounding variables.',
    sampleAnswer:
      "Randomization ensures groups are comparable on all variables (observed and unobserved). **Without randomization**: Selection bias - groups differ systematically. Can't distinguish treatment effect from pre-existing differences. Confounders invalidate results. **Example**: Assigning new model to premium users → can't tell if improvement is due to model or user type. **With randomization**: Differences due to treatment, not confounders. Enables causal inference.",
    keyPoints: [
      'Randomization balances confounders',
      'Without it: selection bias',
      "Can't establish causation without randomization",
      'Groups comparable on all variables',
      'Foundation of causal inference',
    ],
  },
  {
    id: 'q2',
    question:
      'You calculate you need n=2000 per group for 80% power, but only have n=500. What are your options and tradeoffs?',
    hint: 'Consider power, effect size, and type II error.',
    sampleAnswer:
      'With n=500 instead of 2000: **Options**: (1) **Run anyway**: Lower power (~40% instead of 80%) → higher Type II error risk (miss real effect). (2) **Detect larger effect**: n=500 has 80% power for ~6% improvement instead of 3%. (3) **Increase α**: Use α=0.10 instead of 0.05 for more power (but more false positives). (4) **Wait for more data**: Better option if feasible. **Recommendation**: If must run, acknowledge low power. Report negative result as "insufficient evidence" not "no effect." Consider sequential testing.',
    keyPoints: [
      'Low n → low power → miss real effects',
      'Can detect larger effects with same power',
      'Options: accept low power, increase α, wait',
      'Report negative results carefully',
      'Sequential/adaptive designs possible',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the danger of "peeking" at A/B test results before the predetermined sample size is reached?',
    hint: 'Think about multiple testing and false positive rate.',
    sampleAnswer:
      "**Peeking problem**: Each look = hypothesis test. Multiple tests inflate Type I error (false positive rate). **Example**: Test at n=100, 200, 300, 400, 500. Even if no effect, ~20% chance of finding p<0.05 somewhere! **Consequences**: (1) False discoveries, (2) Invalid p-values, (3) Can't trust results. **Solutions**: (1) **Pre-commit**: Decide sample size upfront, test once. (2) **Sequential testing**: Use Bonferroni or specialized methods (e.g., O'Brien-Fleming). (3) **Bayesian**: Probability threshold, not p-values.",
    keyPoints: [
      'Peeking = multiple testing',
      'Inflates false positive rate',
      'Pre-commit to sample size',
      'Test once at end',
      'Sequential testing needs adjustments',
    ],
  },
];
