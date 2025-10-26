export const straddlesStranglesMC = [
  {
    id: 'straddles-strangles-mc-1',
    question: 'A Long Straddle involves buying both an ATM call and an ATM put. What is the breakeven point(s) for this strategy?',
    options: [
      'Only one breakeven at the strike price',
      'Two breakevens: Strike plus total premium, and strike minus total premium',
      'Breakeven at strike price plus premium for calls, strike price only for puts',
      'No breakeven, always profitable if stock moves',
    ],
    correctAnswer: 1,
    explanation:
      'Long Straddle has TWO BREAKEVEN POINTS: Upper breakeven = Strike + Total Premium, Lower breakeven = Strike - Total Premium. Example: $100 strike, $10 total premium → breakevens at $90 and $110. The stock must move MORE than $10 in either direction to profit. Max loss = $10 premium (at $100), max profit = unlimited (if stock moves significantly).',
  },
  {
    id: 'straddles-strangles-mc-2',
    question: 'What is the primary advantage of a Long Strangle compared to a Long Straddle?',
    options: [
      'Higher maximum profit potential',
      'Lower cost to establish (using OTM options instead of ATM)',
      'No theta decay',
      'Guaranteed profit regardless of stock movement',
    ],
    correctAnswer: 1,
    explanation:
      'Long Strangle\'s main advantage is LOWER COST because it uses OTM options instead of ATM. Example: Straddle $10 (ATM), Strangle $6 (OTM). Trade-off: Strangle requires a BIGGER move to profit (wider breakevens). Both strategies suffer from theta decay and are NOT guaranteed profits. Max profit is technically unlimited for both, but strangle gives better percentage returns on large moves due to lower initial cost.',
  },
  {
    id: 'straddles-strangles-mc-3',
    question: 'For an "earnings straddle" play, what is the typical challenge that can lead to losses even if the stock moves significantly?',
    options: [
      'The options expire before earnings are announced',
      'Implied volatility crush after earnings announcement',
      'Dividends are paid reducing option value',
      'The stock always moves exactly to the strike price',
    ],
    correctAnswer: 1,
    explanation:
      'The primary risk for earnings straddles is VOLATILITY CRUSH (IV collapse). Even if the stock moves 10%, the straddle can LOSE value because IV drops 30-50% immediately after earnings. The options were priced with high IV before earnings, and this "fear premium" evaporates post-announcement. Solution: Exit SAME DAY after earnings to capture stock move before full IV decay. Example: Stock moves $15 but straddle loses $5 due to IV dropping from 60% to 35%.',
  },
  {
    id: 'straddles-strangles-mc-4',
    question: 'Why is a "Short Straddle" considered an extremely high-risk strategy?',
    options: [
      'It has limited profit potential capped at the premium received',
      'It has unlimited loss potential on both sides (call and put)',
      'It requires high capital requirements',
      'It can only be traded by professional market makers',
    ],
    correctAnswer: 1,
    explanation:
      'Short Straddle has UNLIMITED LOSS POTENTIAL: Call side: Unlimited if stock soars, Put side: Nearly unlimited if stock crashes (limited only by stock going to $0). Max profit: Premium received (e.g., $10). Risk/Reward is terrible: Collect $10, risk unlimited. Example: Sell $100 straddle for $10. If stock moves to $150, loss = $40 per share. Solution: ALWAYS add protective wings (iron butterfly) to define risk. Never trade naked short straddles.',
  },
  {
    id: 'straddles-strangles-mc-5',
    question: 'In a Long Straddle, which Greek is most favorable, and which is most detrimental to the position?',
    options: [
      'Favorable: Theta (time decay), Detrimental: Vega (volatility)',
      'Favorable: Vega (volatility), Detrimental: Theta (time decay)',
      'Favorable: Delta (direction), Detrimental: Gamma (acceleration)',
      'All Greeks are favorable for Long Straddle',
    ],
    correctAnswer: 1,
    explanation:
      'For Long Straddle: FAVORABLE: Vega (positive vega) - benefits from IV expansion, Gamma (positive gamma) - benefits from large stock moves. DETRIMENTAL: Theta (negative theta) - loses value every day from time decay. Example: $10 straddle loses ~$0.30/day to theta, but gains $0.80 per 1% IV increase. This is why straddles are best bought in LOW IV (cheap options) BEFORE events that will expand volatility.',
  },
];

