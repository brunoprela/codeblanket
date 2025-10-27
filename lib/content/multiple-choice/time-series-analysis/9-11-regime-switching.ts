export const regimeSwitchingMultipleChoice = [
  {
    id: 1,
    question:
      'In a 2-state Markov-switching model with transition probability p=0.95 from state 0 to state 0, what is the expected duration in state 0?',
    options: [
      '0.95 periods',
      '1.05 periods',
      '20 periods',
      '19 periods',
      'Infinite - absorbing state',
    ],
    correctAnswer: 2,
    explanation:
      '20 periods. Expected duration = 1/(1-p) where p is probability of staying. Here: 1/(1-0.95) = 1/0.05 = 20 periods. Interpretation: On average, stay in state 0 for 20 periods before transitioning. If p=0.99 → duration = 100 (very persistent). If p=0.50 → duration = 2 (frequent switching). For regime modeling: High p (>0.95) captures persistent regimes (bull/bear markets last months). Application: Use estimated transition matrix to forecast regime persistence.',
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question: 'Chow test for structural break at known date t* tests:',
    options: [
      'If variance changes at t*',
      'If mean changes at t*',
      'If all regression parameters change at t*',
      'If series becomes non-stationary at t*',
      'If autocorrelation increases after t*',
    ],
    correctAnswer: 2,
    explanation:
      'If ALL regression parameters change. Chow test: Compare models before/after t*. H₀: Same parameters throughout, H₁: Different parameters after break. Method: (1) Fit model on full sample (SSR_all), (2) Fit separately before/after t* (SSR_1 + SSR_2), (3) F-test: F = [(SSR_all - SSR₁ - SSR₂)/k] / [(SSR₁+SSR₂)/(n-2k)] where k = # parameters. Reject if F > critical value. Limitations: Need to know break date, assumes single break. Alternatives: CUSUM for unknown break, Bai-Perron for multiple breaks.',
    difficulty: 'advanced',
  },
  {
    id: 3,
    question:
      'Why is smoothed probability misleading for real-time trading decisions in regime-switching models?',
    options: [
      'Smoothing is less accurate than filtering',
      'Smoothed probabilities use future information (look-ahead bias)',
      'Smoothing assumes normal distribution',
      'Computational cost too high',
      'Smoothing only works for 2 regimes',
    ],
    correctAnswer: 1,
    explanation:
      'Smoothed probabilities use future data. Filtered probability: P(regime_t | data up to t) → available in real-time, Smoothed probability: P(regime_t | ALL data including future) → uses look-ahead, cannot trade on it! Example: At t=100, smoothed prob looks at t=101, 102, ... to better estimate regime at t=100. Backtesting with smoothed probs overstates performance! Always use filtered probabilities for trading signals. Smoothed for: Historical analysis, regime identification for research.',
    difficulty: 'intermediate',
  },
  {
    id: 4,
    question:
      'In regime-switching GARCH, why model regime switches in volatility rather than returns?',
    options: [
      'Returns are harder to model',
      'Volatility shows clearer regime persistence than returns',
      'GARCH only applies to volatility',
      'Regime switches in returns violate EMH',
      'Computational efficiency',
    ],
    correctAnswer: 1,
    explanation:
      'Volatility has clearer regime persistence. Empirical fact: Volatility clusters in persistent regimes (calm 2003-2007, crisis 2008-2009, calm 2017-2019). Returns: Mostly unpredictable, weak persistence. Regime-switching volatility: Captures long-calm/short-crisis patterns better than single-regime GARCH. Model: σ_t² depends on regime s_t where s_t follows Markov chain. Applications: Better VaR estimates, option pricing, risk management. Returns regimes (bull/bear): Less evidence, careful of data mining.',
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'What is the main practical challenge in using regime-switching models for live trading?',
    options: [
      'Parameter estimation is too slow',
      'Cannot compute probabilities in real-time',
      'Regime uncertainty: filtered probabilities never reach 1.0',
      'Models only work with daily data',
      'Require manual regime labeling',
    ],
    correctAnswer: 2,
    explanation:
      'Regime uncertainty - probabilities are probabilistic! Example: P(high vol regime) = 0.75 → 75% confidence, but not certain. Trading implications: (1) Gradual position adjustments (not binary), (2) Risk management across regimes, (3) Robust to misclassification. False positives/negatives: Early detection → costly false signals, Late detection → miss regime change. Solutions: Use confidence thresholds (only act if P > 0.90), combine with other indicators, Monte Carlo for regime uncertainty in backtests.',
    difficulty: 'advanced',
  },
];
