export const coveredCallsProtectivePutsMC = [
  {
    id: 'covered-calls-protective-puts-mc-1',
    question:
      'In a Covered Call strategy, what is the maximum profit potential?',
    options: [
      'Unlimited profit as the stock price rises',
      'The premium received from selling the call option',
      'The strike price minus purchase price, plus the premium received',
      'The stock purchase price plus the premium received',
    ],
    correctAnswer: 2,
    explanation:
      'Maximum profit for a Covered Call = (Strike Price - Stock Purchase Price) + Premium Received. This occurs when the stock closes at or above the strike price at expiration. The stock is called away at the strike, and you keep the premium. Example: Buy stock at $100, sell $105 call for $3 → Max profit = ($105-$100) + $3 = $8 per share. Above $105, profit is capped.',
  },
  {
    id: 'covered-calls-protective-puts-mc-2',
    question:
      'A Protective Put provides downside protection by establishing a "floor" price. Where is this floor located?',
    options: [
      'At the current stock price',
      'At the put strike price minus the premium paid',
      'At the put strike price',
      'At the stock purchase price minus the premium paid',
    ],
    correctAnswer: 1,
    explanation:
      'The floor (maximum loss) for a Protective Put = Put Strike - Stock Purchase Price - Premium Paid. Example: Buy stock at $200, buy $180 put for $5 → Max loss per share = $180 - $200 - $5 = -$25, or floor at $175. The put guarantees you can sell at $180, but you paid $5 premium, so net floor is $175. No matter how low the stock goes (even to $0), your loss is capped at $25/share.',
  },
  {
    id: 'covered-calls-protective-puts-mc-3',
    question:
      'When comparing Covered Calls vs Protective Puts, which statement is most accurate regarding ideal market conditions?',
    options: [
      'Both strategies work best in high implied volatility environments',
      'Covered Calls are best in high IV (sell rich premium), Protective Puts are best in low IV (buy cheap insurance)',
      'Both strategies work best in low implied volatility environments',
      'Covered Calls are best in low IV, Protective Puts are best in high IV',
    ],
    correctAnswer: 1,
    explanation:
      "Covered Calls: BEST in HIGH IV because you're SELLING options - you want to collect inflated premium. High IV Rank (>75%) ideal. Protective Puts: BEST in LOW IV because you're BUYING options - you want cheap insurance. Low IV Rank (<25%) ideal. This is the fundamental principle: sell options when expensive (high IV), buy options when cheap (low IV).",
  },
  {
    id: 'covered-calls-protective-puts-mc-4',
    question:
      'A Collar combines elements of both Covered Calls and Protective Puts. What is the primary advantage of a "zero-cost collar"?',
    options: [
      'No capital is required to establish the position',
      'The premium from the sold call exactly offsets the premium paid for the put, creating no net cost for protection',
      'The position has no risk of loss',
      'The upside potential is unlimited with downside protection',
    ],
    correctAnswer: 1,
    explanation:
      'A zero-cost collar means the PREMIUM from selling the call equals the premium paid for buying the put, resulting in NO NET COST for establishing the hedge. Example: Buy $90 put for $3, sell $110 call for $3, net = $0. You still own the stock (capital required), but the options structure costs nothing. Trade-off: Downside protected at $90, upside capped at $110. This is popular for concentrated stock positions where you want free insurance.',
  },
  {
    id: 'covered-calls-protective-puts-mc-5',
    question:
      'For a monthly income generation strategy using Covered Calls on a $500,000 portfolio, what is a realistic annual return expectation from premiums alone (not including stock appreciation or dividends)?',
    options: [
      '3-5% annually',
      '10-18% annually',
      '25-30% annually',
      '40%+ annually',
    ],
    correctAnswer: 1,
    explanation:
      'Realistic covered call income: 10-18% annually from premiums alone. Breakdown: Selling 30-45 day calls 2-5% OTM generates 1-1.5% monthly premium (12-18% annualized) in favorable conditions (IV Rank > 50%). Lower IV = 0.8-1.2% monthly (10-14% annualized). This is ON TOP of: stock appreciation (capped at strike), dividends (1-2% for dividend stocks). Total returns: 15-25% annually are achievable. 3-5% is too conservative, 25%+ premium alone is unrealistic without high risk.',
  },
];
