export const riskReturnMetricsMC = {
  id: 'risk-return-metrics-mc',
  title: 'Risk and Return Metrics - Multiple Choice',
  questions: [
    {
      id: 'rrm-mc-1',
      type: 'multiple-choice' as const,
      question:
        "A fund has annual return of 15%, total volatility of 20%, and downside deviation (below 0%) of 12%. Risk-free rate is 4%. What is the fund's Sortino ratio?",
      options: ['0.55', '0.75', '0.92', '1.25'],
      correctAnswer: 2,
      explanation:
        'Sortino Ratio = (Return - Risk-Free Rate) / Downside Deviation = (15% - 4%) / 12% = 11% / 12% = 0.92. Answer: C. The Sortino ratio is higher than the Sharpe ratio would be (0.55) because it only penalizes downside volatility, not total volatility. This fund has positive skew - more upside than downside volatility. Sortino ratios >0.80 are considered excellent.',
    },
    {
      id: 'rrm-mc-2',
      type: 'multiple-choice' as const,
      question:
        'Which risk metric is most appropriate for evaluating a tail-hedging strategy that loses money 95% of the time but makes large gains during crashes?',
      options: [
        'Sharpe Ratio',
        'Sortino Ratio',
        'Maximum Drawdown',
        'Omega Ratio',
      ],
      correctAnswer: 3,
      explanation:
        "Answer: D. Omega Ratio captures the full return distribution and is excellent for asymmetric strategies. Sharpe Ratio (A) would show negative values (poor) despite the strategy's valuable insurance properties. Sortino Ratio (B) helps but still misses the extreme positive skewness. Maximum Drawdown (C) measures downside but not the upside benefit. Omega Ratio correctly values strategies that have small frequent losses but large occasional gains.",
    },
    {
      id: 'rrm-mc-3',
      type: 'multiple-choice' as const,
      question:
        'A portfolio peaked at $120 million, dropped to $84 million, then recovered to $110 million. What is the maximum drawdown?',
      options: ['26.7%', '30.0%', '36.0%', '42.9%'],
      correctAnswer: 1,
      explanation:
        'Maximum Drawdown = (Peak - Trough) / Peak = ($120M - $84M) / $120M = $36M / $120M = 0.30 = 30%. Answer: B. The recovery to $110M is irrelevant for maximum drawdown calculation - we only care about the peak-to-trough decline. This 30% drawdown is moderate for an equity portfolio (typical max drawdowns are 30-50% over long periods).',
    },
    {
      id: 'rrm-mc-4',
      type: 'multiple-choice' as const,
      question:
        'An active fund returned 14% while its benchmark returned 11%, with tracking error of 4.5%. What is the Information Ratio?',
      options: ['0.33', '0.67', '0.89', '3.11'],
      correctAnswer: 1,
      explanation:
        'Information Ratio = Active Return / Tracking Error = (14% - 11%) / 4.5% = 3% / 4.5% = 0.67. Answer: B. An IR of 0.67 indicates good active management skill. IR benchmarks: >0.5 is excellent, 0.25-0.5 is good, <0.25 is marginal. The 4.5% tracking error shows moderate active risk, and the 3% active return compensates well for that risk.',
    },
    {
      id: 'rrm-mc-5',
      type: 'multiple-choice' as const,
      question:
        'Which statement about geometric vs arithmetic mean returns is TRUE?',
      options: [
        'Geometric mean is always higher than arithmetic mean',
        'Geometric mean equals arithmetic mean when volatility is zero',
        'Arithmetic mean is more appropriate for long-term wealth projections',
        'The difference between them increases as returns become less volatile',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B. When volatility is zero (constant returns), geometric mean equals arithmetic mean. Statement A is FALSE - geometric mean is always ≤ arithmetic mean. Statement C is FALSE - geometric mean (CAGR) is correct for long-term wealth projections. Statement D is FALSE - the difference increases with MORE volatility, not less. The relationship: Geometric ≈ Arithmetic - (σ²/2), so higher volatility creates larger gap.',
    },
  ],
};
