import { MultipleChoiceQuestion } from '@/lib/types';

export const advancedRiskManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'arm-mc-1',
    question: 'What is CVaR (Conditional VaR)?',
    options: [
      'Same as VaR',
      'Average loss beyond VaR (in worst tail)',
      'Random metric',
      'Ignores risk',
    ],
    correctAnswer: 1,
    explanation:
      'CVaR (Conditional VaR, Expected Shortfall): Average loss in worst X% of cases. If 95% VaR = -$5k, CVaR = average of worst 5% (e.g., -$8k). Better than VaR because captures tail severity. Basel III uses CVaR. More conservative, actionable.',
  },
  {
    id: 'arm-mc-2',
    question: 'What is Sortino ratio?',
    options: [
      'Random metric',
      '(Return - risk_free) / Downside deviation',
      'Price / Volume',
      'Total return',
    ],
    correctAnswer: 1,
    explanation:
      'Sortino: Return per unit of downside risk. Only penalizes losses, not gains. Better than Sharpe for asymmetric returns. Example: Hedge fund with big wins, small losses → Sortino > Sharpe. Used by institutional investors. Formula: (R - Rf) / σ_downside.',
  },
  {
    id: 'arm-mc-3',
    question: 'What is stress testing?',
    options: [
      'Testing software bugs',
      'Simulating portfolio loss in extreme scenarios (crash, vol spike)',
      'Normal conditions only',
      'Ignoring risk',
    ],
    correctAnswer: 1,
    explanation:
      'Stress testing: Simulate extreme scenarios. Examples: -30% market crash, volatility 3x, all correlations → 1. Measures potential loss. If stress loss > 20%, reduce positions. Required by regulators (Dodd-Frank). Protects from tail events (2008, COVID).',
  },
  {
    id: 'arm-mc-4',
    question: 'What is Information Ratio?',
    options: [
      'Random metric',
      'Excess return / Tracking error (measures skill vs benchmark)',
      'Price change',
      'No meaning',
    ],
    correctAnswer: 1,
    explanation:
      'Information Ratio (IR): Alpha / Tracking error. Measures skill beating benchmark per unit of active risk. IR > 0.5 good, > 1.0 excellent. Example: Fund beats S&P by 2% with 3% tracking error → IR = 0.67. Used by institutional investors to evaluate managers.',
  },
  {
    id: 'arm-mc-5',
    question: 'What is Calmar ratio?',
    options: [
      'Random number',
      'Annual return / Maximum drawdown',
      'Price / Earnings',
      'Volatility',
    ],
    correctAnswer: 1,
    explanation:
      'Calmar: Annual return / Max drawdown. Measures return per unit of worst loss. Higher is better. Example: 15% return, 10% max DD → Calmar = 1.5. Good: >1.0. Focuses on drawdown (key for investors). Used alongside Sharpe/Sortino. Hedge funds often target Calmar > 1.',
  },
];
