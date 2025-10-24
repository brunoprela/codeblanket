import { MultipleChoiceQuestion } from '@/lib/types';

export const optionsFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'of-mc-1',
    question:
      'An AAPL call option with strike $150 is trading at $8 when the stock is at $155. What is the time value of this option?',
    options: [
      '$8 (entire premium is time value)',
      '$5 (intrinsic value only)',
      '$3 (premium minus intrinsic value)',
      '$13 (strike plus premium)',
    ],
    correctAnswer: 2,
    explanation:
      "Intrinsic value = max(Stock - Strike, 0) = max(155 - 150, 0) = $5. Time value = Premium - Intrinsic = $8 - $5 = $3. This represents the probability of further price increase before expiration. ATM and slightly ITM options have the highest time value because there's maximum uncertainty about final payoff.",
  },
  {
    id: 'of-mc-2',
    question:
      'You sell a covered call on stock you own at $100, strike $110, premium $3. At expiration, the stock is at $115. What is your total profit per share?',
    options: [
      '$3 (premium only)',
      '$13 (stock gain plus premium)',
      '$10 (strike gain only)',
      '$18 (stock gain to current price)',
    ],
    correctAnswer: 1,
    explanation:
      'Stock gain: $110 - $100 = $10 (capped at strike, stock called away at $110). Premium collected: $3. Total profit: $10 + $3 = $13. Even though stock is at $115, you must sell at $110 (called away). Maximum profit is achieved at strike or higher. This demonstrates the covered call tradeoff: limited upside for premium income.',
  },
  {
    id: 'of-mc-3',
    question:
      'What is the primary advantage of a bull call spread over a long call?',
    options: [
      'Higher maximum profit potential',
      'Unlimited upside if stock rallies',
      'Lower net cost (debit)',
      'No time decay risk',
    ],
    correctAnswer: 2,
    explanation:
      'Bull call spread = buy lower strike call + sell higher strike call. The premium received from selling the higher strike call reduces the net debit paid. Example: Buy $100 call for $5, sell $110 call for $2, net cost $3 (vs $5 for naked long call). Tradeoff: Max profit is capped at spread width minus net debit. Lower cost = better risk/reward ratio but limited upside. Ideal for moderately bullish outlook.',
  },
  {
    id: 'of-mc-4',
    question:
      'When might you exercise an American call option early on a non-dividend paying stock?',
    options: [
      'When the option is deep in-the-money',
      'Just before expiration to capture remaining time value',
      'Never - always sell the option instead',
      'When implied volatility increases',
    ],
    correctAnswer: 2,
    explanation:
      'On non-dividend stocks, American calls should NEVER be exercised early because: (1) Time value is always > 0, (2) Selling the option captures both intrinsic + time value, (3) Exercising captures only intrinsic value (losing time value). Exception: Dividend-paying stocks where dividend > time value. Early exercise before ex-dividend date may be optimal. For puts, early exercise can be optimal if deep ITM to capture intrinsic value now.',
  },
  {
    id: 'of-mc-5',
    question:
      'An iron condor has max profit $400 and max loss $600. What is the breakeven point if the short strikes are $90 (put) and $110 (call)?',
    options: [
      '$90 and $110 (short strikes)',
      '$86 and $114 (wings)',
      '$86 and $114 (short strikes ± net credit)',
      '$90 - $4 = $86 and $110 + $4 = $114',
    ],
    correctAnswer: 3,
    explanation:
      'Iron condor breakevens = Short strikes ± Net credit. Net credit = Max profit = $400 / 100 = $4 per share. Put side breakeven: $90 - $4 = $86. Call side breakeven: $110 + $4 = $114. Between $86-$114, position is profitable. Outside this range, losses begin. Max loss occurs at wings ($600 = spread width - net credit). Example: $10 spread ($90-$80 put, $110-$120 call) - $400 credit = $600 max loss.',
  },
];
