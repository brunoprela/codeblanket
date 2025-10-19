/**
 * Quiz questions for Bayesian Statistics section
 */

export const bayesianstatisticsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the key philosophical difference between Bayesian and Frequentist approaches to statistics.',
    hint: 'Think about how each treats parameters and probability.',
    sampleAnswer:
      '**Frequentist**: Parameters are fixed unknowns. Probability = long-run frequency. CI: "If we repeat procedure, 95% of CIs contain true θ." **Bayesian**: Parameters are random variables. Probability = degree of belief. Credible interval: "95% probability θ is in this interval." Bayesian incorporates prior knowledge; Frequentist uses only data. Both valid but different philosophies.',
    keyPoints: [
      'Frequentist: parameters fixed, probability = frequency',
      'Bayesian: parameters random, probability = belief',
      'Bayesian uses priors + data',
      'Different interpretations of intervals',
      'Both valid approaches',
    ],
  },
  {
    id: 'q2',
    question:
      'How does the prior affect the posterior, and when does its influence diminish?',
    hint: 'Consider strong vs weak priors and amount of data.',
    sampleAnswer:
      'Prior influence depends on: (1) **Prior strength**: Informative (narrow) prior has more influence. Weak/vague prior has little influence. (2) **Data amount**: More data → posterior dominated by likelihood, prior matters less. With infinite data: posterior → MLE regardless of prior. (3) **Prior-data agreement**: If data contradicts strong prior, posterior compromises. **Practical**: Use weak priors if unsure. Prior influence diminishes with more data (good!).',
    keyPoints: [
      'Strong prior → more influence on posterior',
      'More data → prior influence decreases',
      'Infinite data → posterior ≈ MLE',
      'Weak priors for uncertainty',
      'Prior and likelihood combined',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the advantage of Bayesian credible intervals over frequentist confidence intervals for communication to stakeholders?',
    hint: 'Think about what each interval actually means.',
    sampleAnswer:
      '**Credible interval**: "95% probability the parameter is in [a, b]" - direct, intuitive statement. **Confidence interval**: "If we repeated sampling, 95% of intervals would contain the true parameter" - confusing, frequentist interpretation. **For stakeholders**: Credible intervals are what people intuitively want: probability about the parameter. CI requires explaining repeated sampling concept. **Advantage**: Bayesian provides natural probabilistic statements that match how people think.',
    keyPoints: [
      'Credible: direct probability about parameter',
      'Confidence: about procedure, not this interval',
      'Credible intervals more intuitive',
      'Easier to communicate',
      'Matches stakeholder intuition',
    ],
  },
];
