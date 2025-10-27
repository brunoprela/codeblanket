import { MultipleChoiceQuestion } from '@/lib/types';

const transactionCostsAndSlippageQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'tc-slippage-1',
    question:
      'A medium-frequency trading strategy executes 50 round-trip trades per day with an average trade size of 2,000 shares at $100 per share. If the all-in transaction cost is 3 basis points per trade, what is the approximate annual transaction cost?',
    options: ['$456,000', '$912,000', '$1,520,000', '$1,824,000'],
    correctAnswer: 3,
    explanation:
      "Each trade costs: $100 * 2,000 shares * 0.0003 (3 bps) = $60. With 50 round trips per day, that's 100 trades/day (buy and sell) = 100 * $60 = $6,000/day. Over ~252 trading days: $6,000 * 252 = $1,512,000. The closest answer is $1,520,000. This illustrates how seemingly small per-trade costs accumulate dramatically in medium-to-high frequency strategies.",
    difficulty: 'intermediate',
  },
  {
    id: 'tc-slippage-2',
    question:
      'According to the square root market impact model, if you double the size of your order, by approximately what factor does the market impact increase (assuming all other factors constant)?',
    options: [
      'Market impact doubles',
      'Market impact increases by √2 (approximately 1.41x)',
      'Market impact increases by 2√2 (approximately 2.83x)',
      'Market impact quadruples',
    ],
    correctAnswer: 1,
    explanation:
      'The square root law states that impact ∝ √(Q/V), where Q is order size. If Q doubles, √(2Q) = √2 * √Q, so impact increases by approximately 1.41x. This sub-linear relationship is why breaking large orders into smaller pieces can reduce total impact. However, this must be balanced against increased timing risk and opportunity cost.',
    difficulty: 'intermediate',
  },
  {
    id: 'tc-slippage-3',
    question:
      "You're backtesting a high-frequency trading strategy that shows 12% annual returns with zero transaction costs assumed. When you add realistic transaction costs of 2.5 basis points per round trip, the strategy shows -3% returns. What is the most appropriate action?",
    options: [
      'Trade the strategy but with smaller position sizes to reduce costs',
      "Abandon the strategy as it's not viable with realistic costs",
      'Negotiate better commission rates with your broker to make it profitable',
      'Increase leverage to overcome the transaction costs',
    ],
    correctAnswer: 1,
    explanation:
      "A strategy that swings from +12% to -3% with only 2.5 bps of transaction costs is likely trading too frequently or in insufficiently liquid securities. Even with the best possible commission rates (approaching zero for retail), implicit costs (spread, market impact) would remain. Increasing leverage would amplify losses. The strategy fundamentally doesn't work when realistic costs are considered and should be abandoned or significantly modified to trade less frequently.",
    difficulty: 'advanced',
  },
  {
    id: 'tc-slippage-4',
    question:
      'Which of the following is an implicit transaction cost rather than an explicit cost?',
    options: [
      'Broker commission fees',
      'SEC Section 31 fees',
      'Bid-ask spread',
      'Exchange transaction fees',
    ],
    correctAnswer: 2,
    explanation:
      "The bid-ask spread is an implicit cost because it's not a direct fee charged by any party, but rather the cost of immediacy when crossing from one side of the order book to the other. Explicit costs include broker commissions, SEC fees, and exchange fees, which are directly itemized and charged. Implicit costs also include market impact and slippage, which are often larger than explicit costs for institutional traders.",
    difficulty: 'beginner',
  },
  {
    id: 'tc-slippage-5',
    question:
      'A trading desk implements a Transaction Cost Analysis (TCA) system and discovers that their realized slippage is consistently 50% higher than their pre-trade estimates. What is the most comprehensive solution?',
    options: [
      'Immediately recalibrate the slippage model parameters using a rolling 30-day window of actual execution data, implement regime-dependent adjustments, and establish a feedback loop with execution venues',
      'Switch to using limit orders exclusively instead of market orders',
      'Reduce all trade sizes by 50% to compensate for higher slippage',
      "Abandon the strategy as it's no longer profitable",
    ],
    correctAnswer: 0,
    explanation:
      'The comprehensive solution is to calibrate the slippage model using recent execution data, as market microstructure changes over time. A rolling 30-day window captures recent patterns while avoiding overfitting to very short-term noise. Regime-dependent adjustments account for varying market conditions (high/low volatility, different liquidity regimes). A feedback loop with venues helps understand venue-specific costs. Simply switching to limit orders introduces fill risk, reducing size may not address the underlying model inaccuracy, and abandoning the strategy is premature before fixing the cost estimation. Professional trading firms continuously calibrate their TCA models.',
    difficulty: 'advanced',
  },
];

export default transactionCostsAndSlippageQuiz;
