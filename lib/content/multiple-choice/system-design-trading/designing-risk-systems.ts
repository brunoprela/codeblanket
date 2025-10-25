import { MultipleChoiceQuestion } from '@/lib/types';

export const designingRiskSystemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'drs-mc-1',
    question: 'What is the key difference between realized and unrealized P&L?',
    options: [
      'Realized P&L is calculated daily, unrealized is calculated monthly',
      'Realized P&L occurs when positions are closed, unrealized is mark-to-market on open positions',
      'Realized P&L includes commissions, unrealized does not',
      'Realized P&L is for stocks, unrealized is for options',
    ],
    correctAnswer: 1,
    explanation:
      'Realized P&L occurs when position is closed (sold)—actual cash impact. Unrealized P&L is mark-to-market on open positions—paper gains/losses that could disappear. Example: Buy stock @ $100, now $110. Unrealized P&L = +$10/share. Sell @ $110 → realized P&L = +$10/share (now locked in). Both types calculated continuously. Both can include commissions. Applies to all asset types.',
  },
  {
    id: 'drs-mc-2',
    question:
      'Why does Parametric VaR often underestimate risk during financial crises?',
    options: [
      'It requires too much historical data',
      'It assumes normal distribution which has thin tails',
      'It is too computationally expensive',
      'It only works for stock portfolios',
    ],
    correctAnswer: 1,
    explanation:
      'Parametric VaR assumes returns are normally distributed. Normal distribution has thin tails—predicts extreme events are extremely rare. Example: 5 standard deviation move should happen once in 3.5 million years, but happens in markets every few years. During crises: correlations spike, fat tail events occur, normal assumption fails catastrophically. Black Monday 1987 (-22.6%) was 20+ standard deviation event under normal distribution. Historical and Monte Carlo methods better for tail risk.',
  },
  {
    id: 'drs-mc-3',
    question:
      'For an options portfolio, which Greek measures the P&L sensitivity to a 1% change in implied volatility?',
    options: ['Delta', 'Gamma', 'Vega', 'Theta'],
    correctAnswer: 2,
    explanation:
      'Vega measures P&L change per 1% volatility change. Example: Vega = $5,000 means portfolio gains $5,000 if implied volatility increases 1% (e.g., 20% → 21%). Critical for options risk: volatility can spike 50-100%+ during crises. Delta = sensitivity to underlying price. Gamma = sensitivity of delta. Theta = time decay. All Greeks must be monitored for comprehensive options risk management.',
  },
  {
    id: 'drs-mc-4',
    question: 'What is the primary purpose of pre-trade risk checks?',
    options: [
      'To calculate commissions before execution',
      'To prevent orders that would breach risk limits from being submitted',
      'To optimize order routing to exchanges',
      'To calculate expected P&L on the trade',
    ],
    correctAnswer: 1,
    explanation:
      'Pre-trade checks prevent limit breaches before they occur. Check if order would cause: position size > limit, notional > limit, VaR > limit. If projected limit exceeded → reject order immediately. Must be synchronous (<1ms) and block order until check completes. Prevents traders from accidentally (or intentionally) exceeding limits. Commission calculation and routing optimization are separate concerns. P&L estimation useful but secondary to risk prevention.',
  },
  {
    id: 'drs-mc-5',
    question:
      'Why is CVaR (Conditional VaR) considered a better risk measure than VaR for tail risk?',
    options: [
      'CVaR is faster to calculate than VaR',
      'CVaR measures average loss in worst cases beyond VaR threshold',
      'CVaR works better for normal distributions',
      'CVaR requires less historical data than VaR',
    ],
    correctAnswer: 1,
    explanation:
      'CVaR (Expected Shortfall) = average loss when loss exceeds VaR. VaR only tells you threshold, not how bad it gets beyond. Example: 95% VaR = $100K. CVaR might be $150K. Means: in worst 5% of cases, average loss is $150K, not $100K. Important because tail losses often much worse than VaR suggests. CVaR not faster (same computation). Both work with any distribution. CVaR provides more information about catastrophic scenarios than VaR alone.',
  },
];
