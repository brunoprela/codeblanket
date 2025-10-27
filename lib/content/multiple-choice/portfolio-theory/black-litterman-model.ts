export const blackLittermanModelMC = {
  id: 'black-litterman-model-mc',
  title: 'Black-Litterman Model - Multiple Choice',
  questions: [
    {
      id: 'bl-mc-1',
      type: 'multiple-choice' as const,
      question: 'Equilibrium returns in Black-Litterman are calculated using:',
      options: [
        'Historical sample means',
        'Analyst forecasts',
        'Reverse optimization: Π = λΣw_market',
        'CAPM formula',
      ],
      correctAnswer: 2,
      explanation:
        'Answer: C. Black-Litterman uses reverse optimization: if market is optimal, what returns make it optimal? Formula: Π = λΣw_market where λ = risk aversion, Σ = covariance, w_market = market cap weights. This gives stable, forward-looking returns. Not historical means (A - too noisy), not analyst forecasts (B - that\'s the "views"), not CAPM (D - though CAPM and BL equilibrium are related).',
    },
    {
      id: 'bl-mc-2',
      type: 'multiple-choice' as const,
      question:
        'In Black-Litterman, the "confidence" in a view is represented by:',
      options: [
        'A probability between 0 and 1',
        'The Omega (Ω) matrix scaling uncertainty',
        'A t-statistic',
        'The number of analysts agreeing',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B. Confidence is encoded in Omega matrix: Ω = τP∑P^T where τ controls uncertainty. Higher confidence → lower τ → tighter uncertainty → posterior closer to view. Typical: 90% confidence ≈ τ=0.025, 50% confidence ≈ τ=0.5. Not a simple probability (A), not a t-statistic (C), not analyst count (D - though that might inform confidence qualitatively).',
    },
    {
      id: 'bl-mc-3',
      type: 'multiple-choice' as const,
      question:
        'Market cap weights of 70% US, 30% International imply (with λ=2.5, correlations given):',
      options: [
        'US has higher expected return than International',
        'International has higher expected return than US',
        'Expected returns are equal',
        'Cannot determine without volatilities',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B (typically). Reverse optimization: Π = λΣw. International has lower weight (30%) despite likely having similar or higher volatility → market demands higher expected return to hold less of it. The small weight reveals investors require higher return to compensate for perceived higher risk (or lower familiarity). D is wrong because the qualitative answer holds even without exact numbers.',
    },
    {
      id: 'bl-mc-4',
      type: 'multiple-choice' as const,
      question:
        'A Black-Litterman view "US will outperform International by 3%" is what type of view?',
      options: [
        'Absolute view',
        'Relative view',
        'Factor view',
        'Equilibrium view',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B. Relative view expresses one asset vs another: US - Intl = 3%. Absolute view would be "US will return 10%" (no comparison). Relative views are common and often more reliable than absolute predictions. Implementation: P matrix has [1, -1, 0, ...] for US-Intl, Q vector has 3%. Factor views (C) would be based on factors not assets. Equilibrium views (D) aren\'t expressed, they\'re the starting point.',
    },
    {
      id: 'bl-mc-5',
      type: 'multiple-choice' as const,
      question: 'Black-Litterman posterior returns are a weighted average of:',
      options: [
        'Historical returns and analyst forecasts',
        'Equilibrium returns and investor views',
        'Risk-free rate and market return',
        'Long-term and short-term returns',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B. BL combines equilibrium returns (Π from market) with investor views (Q) using Bayesian update. Formula weights by inverse uncertainty: high confidence view gets more weight, low confidence view gets less. Result closer to equilibrium if uncertain, closer to view if confident. Not historical/analyst (A), not CAPM components (C), not time horizons (D). The Bayesian blending is the key BL innovation.',
    },
  ],
};
