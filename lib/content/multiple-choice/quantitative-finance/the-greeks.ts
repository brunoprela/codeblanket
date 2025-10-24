import { MultipleChoiceQuestion } from '@/lib/types';

export const theGreeksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tg-mc-1',
    question:
      'An ATM call option has delta=0.50 and gamma=0.03. If the stock moves up $5, what is the new approximate delta?',
    options: [
      'Delta = 0.50 + 0.03 = 0.53',
      'Delta = 0.50 + (0.03 × 5) = 0.65',
      'Delta = 0.50 × 1.05 = 0.525',
      'Delta stays 0.50 (unchanged)',
    ],
    correctAnswer: 1,
    explanation:
      'Gamma measures how much delta changes per $1 stock move. Delta change = Gamma × Stock change = 0.03 × $5 = 0.15. New delta = 0.50 + 0.15 = 0.65. The option is now deeper in-the-money and behaves more like stock (delta closer to 1.0). Gamma acts like acceleration for delta.',
  },
  {
    id: 'tg-mc-2',
    question:
      'You own a straddle (long call + long put). Which statement about Greeks is TRUE?',
    options: [
      'Delta = +1.0 (directional exposure)',
      'Gamma = 0 (no curvature)',
      'Theta is positive (earn time decay)',
      'Vega is positive (benefit from IV increase)',
    ],
    correctAnswer: 3,
    explanation:
      'Long straddle Greeks: Delta ≈ 0 (call +0.50 and put -0.50 cancel), Gamma > 0 (both options positive gamma, profits from large moves either direction), Theta < 0 (both options lose time value), Vega > 0 (both options benefit from volatility increase). Long options always have positive vega – benefit from IV increases.',
  },
  {
    id: 'tg-mc-3',
    question:
      'An ATM option 30 days to expiration has theta=-$0.10/day. Same option with 5 days to expiration likely has theta closest to:',
    options: [
      '-$0.05/day (lower theta closer to expiration)',
      '-$0.10/day (theta constant)',
      '-$0.30/day (higher theta, accelerated decay)',
      '$0.30/day (positive theta near expiration)',
    ],
    correctAnswer: 2,
    explanation:
      'Theta (time decay) accelerates dramatically as expiration approaches. The curve is non-linear – most time value decays in the final 30 days, especially the last 7 days. An option losing $0.10/day at 30 days might lose $0.30-$0.50/day at 5 days to expiration. This is why holding long options through final week is dangerous unless expecting a major move.',
  },
  {
    id: 'tg-mc-4',
    question: 'Which position has the HIGHEST positive gamma?',
    options: [
      'Long OTM put (strike 20% below stock price)',
      'Long ATM call (strike = stock price)',
      'Long deep ITM call (strike 30% below stock price)',
      'Short ATM call (strike = stock price)',
    ],
    correctAnswer: 1,
    explanation:
      "Gamma is highest for ATM options and decreases as you move OTM or deep ITM. Long options have positive gamma (short options have negative). ATM options have maximum uncertainty about expiration outcome, so delta changes most rapidly with stock moves. Deep ITM options have delta ≈ 1 (won't change much = low gamma). OTM options have low delta that won't change much either.",
  },
  {
    id: 'tg-mc-5',
    question:
      'A portfolio is delta-neutral but has gamma=-50. If the stock moves $10 in either direction, what happens?',
    options: [
      'No change (delta-neutral means no exposure)',
      'Portfolio gains $2,500 from gamma (positive convexity)',
      'Portfolio loses $2,500 from gamma (negative convexity)',
      "Portfolio delta remains zero (gamma doesn't affect delta)",
    ],
    correctAnswer: 2,
    explanation:
      'Gamma P&L = 0.5 × Gamma × (Stock move)² = 0.5 × (-50) × 10² = -$2,500. Negative gamma means losses from large moves in EITHER direction. Even though starting delta-neutral, the position becomes directional as stock moves (delta changes). Short gamma positions (like short straddles, iron condors) lose when stock moves significantly. The loss is quadratic – $20 move causes 4× the loss of $10 move.',
  },
];
