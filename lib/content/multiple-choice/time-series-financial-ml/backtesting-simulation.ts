import { MultipleChoiceQuestion } from '@/lib/types';

export const backtestingSimulationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bs-mc-1',
    question: 'What is lookahead bias?',
    options: [
      'Looking at charts',
      'Using future information in past decisions',
      'Ignoring old data',
      'Trading too fast',
    ],
    correctAnswer: 1,
    explanation:
      'Lookahead bias: Using info not available at decision time. Example: Calculate MA at day 100 but include day 101 price. Inflates backtest returns massively. Prevention: Always shift features by 1 day. Very common bug in backtests.',
  },
  {
    id: 'bs-mc-2',
    question: 'What is survivorship bias?',
    options: [
      'Testing only surviving stocks (ignores bankruptcies)',
      'Testing all stocks',
      'Testing one stock',
      'Testing cryptocurrencies',
    ],
    correctAnswer: 0,
    explanation:
      'Survivorship bias: Backtesting only current index members, excludes delisted/bankrupt stocks (e.g., Enron, Lehman). Inflates returns 2-5%/year. Fix: Use point-in-time universe, include delisted stocks. Critical for accurate backtests.',
  },
  {
    id: 'bs-mc-3',
    question: 'What are realistic transaction costs?',
    options: [
      'Zero costs',
      '0.1% commission + 0.05% slippage',
      '50% commission',
      'Free trading',
    ],
    correctAnswer: 1,
    explanation:
      'Realistic: Commission 0.05-0.1% (retail) or 0.01% (institutional), Slippage 0.05-0.1% (market orders). High-frequency: costs higher. Total: 0.2-0.3% round-trip. Strategies with 1000 trades/year lose 2-3% to costs. Critical to include.',
  },
  {
    id: 'bs-mc-4',
    question: 'What is a good Sharpe ratio for a trading strategy?',
    options: ['>0.5', '>1.0', '>5.0', '>10.0'],
    correctAnswer: 1,
    explanation:
      'Sharpe > 1.0 is good, > 1.5 excellent, > 2.0 exceptional. SPY Sharpe ≈ 0.5. HFT strategies can hit 2-3. Sharpe > 3 likely overfit or lookahead bias. In-sample Sharpe often 2x out-of-sample. Target: 1.5 in-sample → 0.75-1.0 out-of-sample.',
  },
  {
    id: 'bs-mc-5',
    question: 'What is Monte Carlo simulation in backtesting?',
    options: [
      'Drive to Monaco',
      'Resample trades randomly to test robustness',
      'Use one backtest',
      'Ignore uncertainty',
    ],
    correctAnswer: 1,
    explanation:
      'Monte Carlo: Resample trades with replacement 1000 times, calculate return distribution. Check if strategy robust (consistent) or fragile (few lucky trades). Good: 95% of simulations positive. Bad: Highly variable, depends on few outliers. Essential robustness check.',
  },
];
