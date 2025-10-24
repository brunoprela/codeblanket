export const portfolioOptimizationQuiz = [
  {
    id: 'po-q-1',
    question:
      'Explain Markowitz mean-variance optimization and its limitations in practice.',
    sampleAnswer:
      'Markowitz: Maximize Sharpe ratio (return/volatility) subject to weights summing to 1. Finds optimal risk-return tradeoff. Limitations: (1) Sensitive to input estimates (GIGO), (2) Assumes normal returns (fat tails in reality), (3) Static (one period), (4) Transaction costs ignored, (5) Concentrated portfolios. Solutions: Robust optimization, Black-Litterman, regularization, constraints.',
    keyPoints: [
      'Maximize Sharpe = return/volatility',
      'Efficient frontier: min vol for each return level',
      'Limitations: estimation error, concentrated portfolios',
      'Solutions: shrinkage, Black-Litterman, constraints',
      'Practical: Add min/max weight constraints (5%-40%)',
    ],
  },
  {
    id: 'po-q-2',
    question: 'What is risk parity? When is it better than maximum Sharpe?',
    sampleAnswer:
      'Risk Parity: Each asset contributes equal risk (not equal weight). If asset A has 2x volatility of B, give B 2x weight. Better than max Sharpe when: (1) Diversification important, (2) Estimation uncertainty high, (3) Long-term horizon. Max Sharpe can be concentrated (70% in one asset), risk parity always balanced. Used by Bridgewater All Weather fund.',
    keyPoints: [
      'Equal risk contribution, not equal weight',
      'High-vol assets get lower weight',
      'More diversified than max Sharpe',
      'Robust to estimation errors',
      'Popular for multi-asset portfolios',
    ],
  },
  {
    id: 'po-q-3',
    question:
      'How do you incorporate views/predictions into portfolio optimization?',
    sampleAnswer:
      'Black-Litterman model: Combine market equilibrium (CAPM) with investor views. Start with market weights, adjust based on predicted outperformance. Example: Predict AAPL +15% next year with 80% confidence. BL blends this with market equilibrium. More confident views get higher weight. Avoids estimation error of pure Markowitz. Widely used by institutional investors.',
    keyPoints: [
      'Black-Litterman: equilibrium + views',
      'Confidence level determines view weight',
      'Prevents extreme portfolios from noisy estimates',
      'Example: bullish on tech → overweight tech',
      'More robust than using predictions directly',
    ],
  },
];
