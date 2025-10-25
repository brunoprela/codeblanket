import { MultipleChoiceQuestion } from '@/lib/types';

export const derivativesOverviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-3-mc-1',
    question:
      'A company will receive €10M in 6 months. Current spot: $1.10/€, 6-month forward: $1.08/€. They enter a forward to sell €10M at $1.08. In 6 months, spot is $1.15. What is their effective USD amount received?',
    options: [
      '$10.8M (locked in forward rate)',
      '$11.5M (spot rate)',
      '$11.25M (average)',
      'Depends on counterparty',
    ],
    correctAnswer: 0,
    explanation:
      'Forward contract locks in rate at $1.08/€ regardless of future spot. They receive exactly $10.8M. They miss the upside of EUR strengthening to $1.15 (which would have been $11.5M), but they have certainty. This is the trade-off: forwards eliminate uncertainty but also eliminate upside.',
  },
  {
    id: 'fm-1-3-mc-2',
    question:
      'What is the key difference between futures and forwards regarding daily cash flows?',
    options: [
      'Futures have no cash flows until maturity',
      'Futures are marked-to-market daily with cash settlement',
      'Forwards require daily margin calls',
      'There is no difference',
    ],
    correctAnswer: 1,
    explanation:
      'Futures: Daily mark-to-market. Each day, P&L is settled in cash. Example: Long futures at 4000, closes at 4010 → receive $500 that day. Next day starts at 4010. Forwards: No cash flows until maturity. All P&L settles at expiry. This makes futures better for speculation (daily liquidity) but forwards better for hedging (no margin hassle).',
  },
  {
    id: 'fm-1-3-mc-3',
    question:
      'An at-the-money call option (strike = spot = $100) has delta = 0.50. If the stock rises to $101, approximately what is the option price increase?',
    options: ['$0.25', '$0.50', '$0.75', '$1.00'],
    correctAnswer: 1,
    explanation:
      'Delta = 0.50 means option price changes $0.50 for every $1 stock move. Stock up $1 → option up ~$0.50. This is an approximation (exact change depends on gamma). Delta hedging: To neutralize, short 0.50 shares per option. ATM options have delta ≈ 0.50, deep ITM ≈ 1.0, deep OTM ≈ 0.',
  },
  {
    id: 'fm-1-3-mc-4',
    question:
      'A trader is long 100 call options with delta = 0.60 each. To delta-hedge the position, they should:',
    options: [
      'Buy 60 shares',
      'Short 60 shares',
      'Buy 100 shares',
      'Short 100 shares',
    ],
    correctAnswer: 1,
    explanation:
      'Portfolio delta = 100 options × 0.60 delta = 60. To neutralize delta (delta-neutral), short 60 shares. If stock rises $1: options gain ~$60, shares lose $60, net zero. Note: Gamma means delta changes, so hedge needs rebalancing. Market makers delta-hedge continuously to remain neutral.',
  },
  {
    id: 'fm-1-3-mc-5',
    question:
      'An option has vega = 0.15 and current volatility = 25%. If volatility rises to 30%, what is the approximate option price increase?',
    options: ['$0.15', '$0.45', '$0.75', '$1.50'],
    correctAnswer: 2,
    explanation:
      'Vega = 0.15 means option gains $0.15 per 1% volatility increase. Vol rises 5% (from 25% to 30%) → option gains 5 × $0.15 = $0.75. Long options have positive vega (benefit from vol increase), short options negative vega (hurt by vol increase). Vega is highest for ATM options near expiry.',
  },
];
