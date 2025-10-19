/**
 * Quiz questions for Common Continuous Distributions section
 */

export const commoncontinuousdistributionsQuiz = [
  {
    id: 'q1',
    question:
      "The Exponential distribution has a memoryless property, just like the Geometric distribution. Explain this property and discuss when it's appropriate vs inappropriate to assume memorylessness in machine learning applications.",
    hint: 'Think about whether past events influence future probabilities.',
    sampleAnswer:
      "The memoryless property states that P(X > s+t | X > s) = P(X > t) - the probability of waiting additional time t doesn't depend on how long you've already waited (s). For exponential: if you've waited 5 minutes for a bus, the expected additional wait is still 1/λ, same as if you just arrived. This arises from modeling truly independent events at constant rate. Appropriate uses: (1) Radioactive decay - each atom decays independently. (2) Server requests when clients are independent. (3) Phone calls arriving at call center (roughly). Inappropriate uses: (1) Learning algorithms - past attempts improve future success probability, violating memorylessness. (2) Bus arrivals on schedule - if 10 minutes late, bus is likely coming soon. (3) Machine failures - wear and tear creates memory. (4) User behavior - past actions influence future actions. In ML: Use exponential for truly random events (Poisson process), not for systems with memory/learning/dependency. Test assumption by checking if P(X > 10 | X > 5) ≈ P(X > 5).",
    keyPoints: [
      'Memoryless: P(X > s+t | X > s) = P(X > t)',
      "Past doesn't affect future for exponential processes",
      'Appropriate for independent events at constant rate',
      'Inappropriate for learning, wear, or scheduled systems',
      'Critical assumption to verify before using exponential model',
    ],
  },
  {
    id: 'q2',
    question:
      "The Beta distribution is a conjugate prior for the Bernoulli distribution. Explain what this means and why it's computationally valuable in Bayesian machine learning.",
    sampleAnswer:
      'A conjugate prior means: if your prior is Beta and your likelihood is Bernoulli/Binomial, then your posterior is also Beta. Mathematically: Prior Beta(α, β) + Binomial likelihood with k successes in n trials → Posterior Beta(α+k, β+n-k). This is valuable because: (1) Closed-form updates - no need for numerical integration or MCMC. Just add observed successes to α and failures to β. (2) Interpretable parameters - α acts like "prior successes", β like "prior failures". (3) Sequential updates - can update belief as data arrives. (4) Computationally fast - just parameter updates, not recomputing integrals. Example: A/B testing. Start with Beta(1,1) prior (uniform). Observe 12 conversions in 100 visitors. Posterior is Beta(1+12, 1+88) = Beta(13, 89). Mean conversion rate = 13/(13+89) = 12.7%. Can immediately compute credible intervals. Without conjugacy, we\'d need expensive numerical methods. Conjugacy is why Beta is THE choice for modeling probabilities in Bayesian ML.',
    keyPoints: [
      'Conjugate prior: posterior has same family as prior',
      'Beta + Bernoulli/Binomial → Beta posterior',
      'Closed-form updates: no numerical integration',
      'Interpretable: α = successes, β = failures',
      'Computationally efficient for Bayesian inference',
    ],
  },
  {
    id: 'q3',
    question:
      'Why does the t-distribution have heavier tails than the normal distribution, and why does this matter for statistical inference and machine learning?',
    sampleAnswer:
      'The t-distribution has heavier tails because it accounts for uncertainty in the sample variance. With small samples, we don\'t know the true population variance, so we estimate it from data. This additional uncertainty manifests as heavier tails - more extreme values are possible. Mathematically: t = (X̄ - μ)/(s/√n) where s is sample std dev. As degrees of freedom ν → ∞, s → σ (true std dev) and t → normal. Why it matters: (1) Small samples - with n<30, use t-distribution for confidence intervals, not normal. Using normal would be overconfident (too narrow intervals). (2) Outliers - heavier tails make t-distribution more robust. Extreme values are less "surprising". (3) Hypothesis testing - t-tests account for estimation uncertainty. (4) ML robustness - can use t-distribution as error model instead of Gaussian for outlier-robust regression. (5) Bayesian inference - t-distribution priors are more robust than Gaussian. Practical: Always use t-distribution when σ is unknown and n is small (<30). Heavier tails = honest representation of uncertainty.',
    keyPoints: [
      'Heavier tails from uncertainty in sample variance',
      'More extreme values possible with small samples',
      'Use t-distribution when σ unknown and n < 30',
      'More robust to outliers than normal',
      'Honest representation of uncertainty',
    ],
  },
];
