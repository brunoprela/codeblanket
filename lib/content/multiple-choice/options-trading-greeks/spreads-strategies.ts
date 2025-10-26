export const spreadsStrategiesMC = [
  {
    id: 'spreads-strategies-mc-1',
    question: 'What is the primary difference between a Bull Call Spread and a Bull Put Spread in terms of cash flow?',
    options: [
      'Both require paying a debit',
      'Bull Call Spread requires a debit (pay), Bull Put Spread generates a credit (receive)',
      'Bull Call Spread generates a credit, Bull Put Spread requires a debit',
      'Neither requires any capital since they are spreads',
    ],
    correctAnswer: 1,
    explanation:
      'Bull Call Spread: Buy lower strike call, sell higher strike call → Net DEBIT (you pay more for long than receive for short). Bull Put Spread: Sell higher strike put, buy lower strike put → Net CREDIT (you receive more for short than pay for long). Both are bullish, but credit spreads (bull put) are preferred in HIGH IV, debit spreads (bull call) in LOW IV.',
  },
  {
    id: 'spreads-strategies-mc-2',
    question: 'In a Long Call Butterfly spread, where is the maximum profit achieved?',
    options: [
      'When the stock is at or below the lowest strike',
      'When the stock is exactly at the middle strike',
      'When the stock is at or above the highest strike',
      'At any price between the lowest and highest strikes',
    ],
    correctAnswer: 1,
    explanation:
      'A Long Call Butterfly (buy 1 low, sell 2 middle, buy 1 high) achieves MAXIMUM PROFIT when the stock closes EXACTLY AT THE MIDDLE STRIKE at expiration. The profit zone is narrow around the middle strike. Max loss occurs if stock is at/below the lowest strike or at/above the highest strike (both outside wings). This is a neutral strategy betting on minimal movement.',
  },
  {
    id: 'spreads-strategies-mc-3',
    question: 'What is the key advantage of an Iron Condor compared to an Iron Butterfly?',
    options: [
      'Higher maximum profit potential',
      'Wider profit range and higher probability of profit',
      'Lower capital requirement',
      'Unlimited profit potential on one side',
    ],
    correctAnswer: 1,
    explanation:
      'Iron Condor has a WIDER PROFIT RANGE than Iron Butterfly, leading to HIGHER PROBABILITY OF PROFIT. Iron Butterfly: Tight range around ATM ($95-$105), ~40% prob profit. Iron Condor: Wider range ($93-$107), ~65% prob profit. Trade-off: Condor collects less premium (lower max profit) but wins more often. Butterfly: higher credit but must be very precise.',
  },
  {
    id: 'spreads-strategies-mc-4',
    question: 'In a Calendar Spread (Time Spread), what is the primary source of profit?',
    options: [
      'The stock making a large directional move',
      'Implied volatility decreasing significantly',
      'The front-month option decaying faster than the back-month option (theta)',
      'Assignment on the short option',
    ],
    correctAnswer: 2,
    explanation:
      'Calendar Spread profits primarily from THETA DECAY: the front-month (short) option loses time value faster than the back-month (long) option. This creates profit if the stock stays near the strike. Calendar spreads work best with STABLE stocks (minimal movement). They also benefit from IV expansion (positive vega), but theta is the main driver. Large moves hurt calendar spreads.',
  },
  {
    id: 'spreads-strategies-mc-5',
    question: 'For a Bear Call Spread, what is the maximum loss scenario?',
    options: [
      'Stock closes at or below the lower (short) strike',
      'Stock closes between the two strikes',
      'Stock closes at or above the higher (long) strike',
      'Stock remains exactly at the current price',
    ],
    correctAnswer: 2,
    explanation:
      'Bear Call Spread (sell lower strike call, buy higher strike call) has MAX LOSS when stock closes AT OR ABOVE THE HIGHER STRIKE. Max Loss = (Higher Strike - Lower Strike) - Credit Received. Example: Sell 100 call, buy 105 call, receive $2 credit → max loss = $5 - $2 = $3 (at $105+). Max profit = $2 credit (at $100 or below). This is a bearish/neutral strategy for credit collection.',
  },
];

