/**
 * Quiz questions for Maximum Likelihood Estimation section
 */

export const maximumlikelihoodestimationQuiz = [
  {
    id: 'q1',
    question:
      'Explain why we use log-likelihood instead of likelihood in practice. What numerical problems does this solve?',
    hint: 'Think about multiplying many small probabilities.',
    sampleAnswer:
      'Log-likelihood solves numerical underflow: likelihood = ∏P(x_i) multiplies many small probabilities (<1), causing underflow (→0). Log converts products to sums: log(∏P) = ΣlogP, which is numerically stable. Also easier to optimize (derivatives simpler). Used universally in ML for stability.',
    keyPoints: [
      'Products of probabilities cause underflow',
      'Log converts products to sums',
      'Numerically stable',
      'Easier optimization',
      'Monotonic transformation preserves maxima',
    ],
  },
  {
    id: 'q2',
    question:
      'How is training a neural network with cross-entropy loss related to maximum likelihood estimation?',
    hint: 'Think about what cross-entropy represents.',
    sampleAnswer:
      'Cross-entropy = negative log-likelihood for classification. For binary: CE = -Σ[y·log(p) + (1-y)·log(1-p)] is NLL of Bernoulli. Minimizing CE = maximizing likelihood! Training neural nets = MLE. This explains why CE is the "right" loss for classification - it\'s the principled statistical approach.',
    keyPoints: [
      'Cross-entropy = negative log-likelihood',
      'Minimizing loss = maximizing likelihood',
      'Neural network training = MLE',
      'Explains why CE is standard for classification',
      'Statistical foundation for deep learning',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is OLS (ordinary least squares) equivalent to MLE under the assumption of normal errors?',
    hint: 'Write out the log-likelihood for normal distribution.',
    sampleAnswer:
      'Under normality, errors ~ N(0,σ²). Log-likelihood = -n/2·log(2πσ²) - Σ(y-ŷ)²/(2σ²). Maximizing LL = minimizing Σ(y-ŷ)² (the RSS). This is exactly OLS! Conclusion: OLS is MLE when errors are normal. Without normality, MLE and OLS differ. This justifies OLS from statistical principles.',
    keyPoints: [
      'Normal errors: LL = -constant - Σ(y-ŷ)²/(2σ²)',
      'Maximizing LL = minimizing squared errors',
      'OLS = MLE under normality',
      'Provides statistical justification for OLS',
      'Different distributions → different estimators',
    ],
  },
];
