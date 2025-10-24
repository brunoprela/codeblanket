import { MultipleChoiceQuestion } from '@/lib/types';

export const portfolioOptimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'po-mc-1',
    question: 'What is the Sharpe ratio?',
    options: [
      'Return / Price',
      '(Return - Risk-free rate) / Volatility',
      'Volatility / Return',
      'Maximum drawdown',
    ],
    correctAnswer: 1,
    explanation:
      'Sharpe = (Return - Risk-free rate) / Volatility. Measures risk-adjusted returns. Higher is better. Sharpe > 1 is good, > 2 is excellent. Example: 12% return, 2% risk-free, 15% vol → Sharpe = (0.12-0.02)/0.15 = 0.67.',
  },
  {
    id: 'po-mc-2',
    question: 'What is the efficient frontier?',
    options: [
      'The edge of the office',
      'Set of portfolios with max return for each risk level',
      'Highest return portfolio only',
      'Lowest risk portfolio only',
    ],
    correctAnswer: 1,
    explanation:
      'Efficient frontier: Curve showing maximum return for each volatility level. All portfolios below the curve are sub-optimal. Optimal portfolio depends on risk tolerance. Conservative → low vol, aggressive → high vol. Max Sharpe portfolio is tangent point.',
  },
  {
    id: 'po-mc-3',
    question: 'What is risk parity?',
    options: [
      'Equal weights for all assets',
      'Equal risk contribution from each asset',
      'Zero risk',
      'Maximum risk',
    ],
    correctAnswer: 1,
    explanation:
      'Risk parity: Each asset contributes equal % of portfolio risk. If stocks 2x vol of bonds, bonds get 2x weight. Result: ~60% bonds, 40% stocks (typical 60/40 reversed). More diversified than equal weight. Used by Bridgewater All Weather.',
  },
  {
    id: 'po-mc-4',
    question: 'Why add minimum weight constraints (e.g., each asset ≥ 5%)?',
    options: [
      'Looks nicer',
      'Prevents over-concentration from estimation error',
      'Increases transaction costs',
      'Guarantees higher returns',
    ],
    correctAnswer: 1,
    explanation:
      'Unconstrained optimization often gives extreme portfolios (80% in one asset) due to small estimation errors. Min/max constraints (e.g., 5%-40%) force diversification, reduce sensitivity to inputs, more robust. Slight Sharpe decrease but better real-world performance.',
  },
  {
    id: 'po-mc-5',
    question: 'What is Black-Litterman model?',
    options: [
      'A color optimization algorithm',
      'Combines market equilibrium with investor views',
      'Only uses historical data',
      'Ignores risk',
    ],
    correctAnswer: 1,
    explanation:
      'Black-Litterman: Start with market equilibrium (CAPM), adjust with views. "I think AAPL outperforms 10% with 80% confidence." Blends views with market. Prevents extreme portfolios from noisy predictions. More robust than pure Markowitz. Used by institutional investors.',
  },
];
