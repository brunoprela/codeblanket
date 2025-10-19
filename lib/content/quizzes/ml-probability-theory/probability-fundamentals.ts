/**
 * Quiz questions for Probability Fundamentals section
 */

export const probabilityfundamentalsQuiz = [
  {
    id: 'q1',
    question:
      "Explain why probability theory is fundamental to machine learning. How does uncertainty manifest in ML systems, and why can't we simply use deterministic algorithms?",
    hint: 'Think about real-world data, training processes, and prediction confidence.',
    sampleAnswer:
      'Probability theory is essential to machine learning because uncertainty is inherent at every level: (1) Data uncertainty - real-world data is noisy, incomplete, and contains measurement errors; we never have perfect information. (2) Model uncertainty - we never know the true underlying function; our models are approximations with limited capacity. (3) Prediction uncertainty - even with the best model, we can rarely be 100% certain about outcomes. (4) Training uncertainty - stochastic optimization uses random mini-batches, initialization is random, and dropout randomly zeros neurons. Deterministic algorithms would be too brittle and couldn\'t handle the inherent randomness in data or provide confidence measures. Probability allows us to quantify "how confident" we are in predictions, which is critical for high-stakes decisions like medical diagnosis or autonomous driving.',
    keyPoints: [
      'Data is inherently noisy and incomplete',
      'Models are approximations with uncertainty',
      'Training involves randomness (SGD, initialization, dropout)',
      'Probabilistic predictions provide confidence measures',
      'Allows principled decision-making under uncertainty',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk through the three Kolmogorov axioms of probability and explain why each one is necessary. What would break if we violated any of them?',
    sampleAnswer:
      "The three Kolmogorov axioms are: (1) Non-negativity: P(E) ≥ 0. This is necessary because probability represents likelihood - negative probabilities are meaningless. If violated, we couldn't interpret probabilities as proportions or frequencies. (2) Normalization: P(Ω) = 1. This ensures that something must happen - the total probability of all outcomes is 100%. If violated, probabilities wouldn't be comparable or interpretable. (3) Additivity: For disjoint events A and B, P(A∪B) = P(A) + P(B). This ensures consistency - if events can't happen simultaneously, their combined probability is just the sum. If violated, we'd get paradoxes where probabilities don't add up correctly. Together, these axioms ensure probability is a coherent mathematical framework. They're the foundation that allows us to derive all other probability rules (complement, addition, multiplication rules).",
    keyPoints: [
      'Non-negativity: Probabilities cannot be negative',
      'Normalization: Total probability must equal 1',
      'Additivity: Disjoint events probabilities sum',
      'Violations lead to mathematical inconsistencies',
      'These axioms enable derivation of all other rules',
    ],
  },
  {
    id: 'q3',
    question:
      'In machine learning, we often use empirical (frequentist) probability estimates from data. Explain the relationship between empirical probability and theoretical probability. What happens as we collect more data?',
    hint: 'Think about the Law of Large Numbers.',
    sampleAnswer:
      'Empirical probability is calculated from observed data: P(E) ≈ (count of E) / (total trials). Theoretical probability is the "true" probability based on the underlying process. The Law of Large Numbers guarantees that as we collect more data, empirical probability converges to theoretical probability. With 10 samples, our estimate might be way off (e.g., 7 heads in 10 flips = 70% vs true 50%). With 10,000 samples, we\'d get very close to 50%. This is why: (1) More data always helps in ML - better probability estimates. (2) Small datasets lead to overconfident or inaccurate models. (3) We need sufficient data to estimate rare events. (4) Cross-validation works because we\'re estimating true generalization performance. In practice, we never have infinite data, so we use techniques like regularization (Bayesian priors) to compensate for limited samples.',
    keyPoints: [
      'Empirical probability: estimated from observed data',
      'Theoretical probability: true underlying probability',
      'Law of Large Numbers: empirical → theoretical as n → ∞',
      'More data leads to better probability estimates',
      'Small datasets lead to poor estimates and overfitting',
    ],
  },
];
