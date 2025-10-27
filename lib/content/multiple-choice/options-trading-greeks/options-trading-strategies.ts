export const optionsTradingStrategiesMC = [
  {
    id: 'options-trading-strategies-mc-1',
    question:
      'A Bull Call Spread involves buying a lower strike call and selling a higher strike call. What is the primary advantage compared to buying a naked call?',
    options: [
      'Unlimited profit potential instead of capped profit',
      'Lower capital requirement due to premium received from sold call offsetting cost',
      'Positive theta (time decay works in your favor)',
      'No risk of assignment on the long call position',
    ],
    correctAnswer: 1,
    explanation:
      'The primary advantage of a bull call spread vs naked call is LOWER CAPITAL REQUIREMENT. By selling the higher strike call, you receive premium that offsets the cost of the long call (e.g., buy 100 call for $5, sell 105 call for $2, net debit = $3 instead of $5). The trade-off is capped upside (profit limited to spread width minus debit). Theta is still negative overall, and assignment risk exists on the short call if ITM.',
  },
  {
    id: 'options-trading-strategies-mc-2',
    question:
      'An Iron Condor is constructed by selling an OTM put, buying a further OTM put, selling an OTM call, and buying a further OTM call. This strategy profits primarily when:',
    options: [
      'The stock makes a large directional move in either direction',
      'Implied volatility expands significantly',
      'The stock stays within a range (between the sold strikes) and time decays',
      'The stock trends strongly in one direction with low volatility',
    ],
    correctAnswer: 2,
    explanation:
      'An Iron Condor profits when the stock STAYS WITHIN THE RANGE defined by the sold strikes (the two middle strikes). The strategy has positive theta (benefits from time decay) and negative vega (benefits from IV contraction or stability). Max profit is achieved if the stock closes between the sold put and sold call at expiration. Large moves in either direction result in losses (capped by the wings).',
  },
  {
    id: 'options-trading-strategies-mc-3',
    question:
      'When comparing a Short Put to a Bull Call Spread for a bullish outlook, the Short Put typically requires:',
    options: [
      'Much less capital because you receive a credit',
      "No margin or collateral since it's a credit trade",
      'Significantly more capital (cash-secured or margin) but has positive theta',
      'The same capital as the Bull Call Spread',
    ],
    correctAnswer: 2,
    explanation:
      'A Short Put requires SIGNIFICANTLY MORE CAPITAL than a Bull Call Spread. Example: Sell 95 put = $9,500 cash-secured (or $2,000-3,000 margin), while Bull Call Spread 100/105 = $300 debit. The short put does have positive theta (time decay benefit) and collects premium, but the capital requirement is 10-30× higher. This makes it suitable for accounts with more capital and those comfortable with potential assignment.',
  },
  {
    id: 'options-trading-strategies-mc-4',
    question:
      'A Calendar Spread (Time Spread) involves selling a near-term option and buying a longer-term option at the same strike. This strategy profits primarily from:',
    options: [
      'Large directional moves in the underlying stock',
      'Rapid expansion of implied volatility in both options',
      'Theta decay of the near-term option while the long-term option retains value',
      'Assignment on the short option forcing early exercise',
    ],
    correctAnswer: 2,
    explanation:
      'A Calendar Spread profits primarily from THETA DECAY of the near-term (short) option, while the longer-term (long) option retains more of its value. The strategy works best when the stock stays near the strike price (typically ATM). If the stock moves significantly away, both options lose value. Calendar spreads benefit modestly from IV expansion (positive vega) and are often used in neutral to slightly bullish/bearish outlooks.',
  },
  {
    id: 'options-trading-strategies-mc-5',
    question:
      'In a high implied volatility environment (IV Rank > 75%), which type of options strategy is generally most favorable?',
    options: [
      'Buying naked calls and puts to benefit from inflated premium',
      'Long straddles to profit from expected large moves',
      'Selling premium strategies like short strangles or iron condors',
      'Calendar spreads that buy expensive long-term options',
    ],
    correctAnswer: 2,
    explanation:
      "In HIGH IV environments (IV Rank > 75%), SELLING PREMIUM is most favorable (short strangles, iron condors, credit spreads). High IV means options are expensive, so you collect inflated premium. These strategies have positive theta (time decay) and negative vega (benefit when IV inevitably contracts back to the mean). Buying options in high IV is generally unfavorable because you're paying elevated prices, and IV contraction will hurt the position even if you're directionally correct.",
  },
];
