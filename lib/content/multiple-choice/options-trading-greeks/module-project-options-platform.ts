export const moduleProjectOptionsPlatformMC = [
  {
    id: 'module-project-options-platform-mc-1',
    question:
      'In implementing a Black-Scholes pricing engine, what is the most critical edge case to handle?',
    options: [
      'When the stock price equals the strike price exactly',
      'When time to expiration approaches zero (T → 0)',
      'When volatility is exactly 50%',
      'When the risk-free rate is negative',
    ],
    correctAnswer: 1,
    explanation:
      'When T → 0 (near expiration), Black-Scholes formula breaks down: d1 and d2 approach infinity/undefined, vega → 0 (no more time value), price converges to intrinsic value. Must handle specially: if T < 0.01 (or some threshold), return intrinsic value directly: max(S-K, 0) for calls, max(K-S, 0) for puts, set all Greeks to limiting values (delta = 1 or 0, gamma/theta/vega = 0). This prevents numerical errors and division by zero in calculations.',
  },
  {
    id: 'module-project-options-platform-mc-2',
    question:
      'When building a Strategy Builder for options, what is the most important consideration for calculating aggregate portfolio Greeks?',
    options: [
      'Using the most recent stock price only',
      'Properly accounting for the sign (long vs short) and multiplier (contracts × 100) of each leg',
      'Only calculating delta, ignoring other Greeks',
      'Assuming all options have the same expiration',
    ],
    correctAnswer: 1,
    explanation:
      'Aggregate Greeks MUST account for: Sign: Long option (+delta, +vega), short option (-delta, -vega), Multiplier: Each contract = 100 shares, so multiply by quantity × 100. Example: Long 10 calls (delta 0.50 each) = +10 × 100 × 0.50 = +500 portfolio delta. Short 5 puts (delta -0.30) = -5 × 100 × (-0.30) = +150 delta. Total portfolio delta = +650. Incorrect accounting leads to completely wrong risk measures.',
  },
  {
    id: 'module-project-options-platform-mc-3',
    question:
      'In a Portfolio Manager risk monitoring system, what should trigger an immediate hedge action?',
    options: [
      'Any negative daily P&L',
      'Portfolio Greeks exceeding RED limit thresholds (e.g., |delta| > 10,000)',
      'Stock price moving more than 1%',
      'Any new position being added',
    ],
    correctAnswer: 1,
    explanation:
      'Immediate hedge triggered by: RED LIMIT BREACH (e.g., |delta| > 10,000, |vega| > $50K). Action: Auto-generate hedge recommendation, Alert risk manager (SMS/email), Execute hedge (if approved). Example: Delta = +12,000 (exceeds limit 10,000) → recommend: "Sell 7,000 SPY shares to reduce delta to +5,000". Orange alerts (80% of limit) warn but don\'t require immediate action. This is how professional risk management works - set limits in advance, enforce strictly.',
  },
  {
    id: 'module-project-options-platform-mc-4',
    question:
      'In a Backtesting Framework, what is the most realistic way to model option entry/exit prices?',
    options: [
      'Use the Black-Scholes theoretical mid-price',
      'Use the mid-point between bid and ask, plus slippage and commission',
      'Use the bid price for all trades',
      'Assume zero transaction costs',
    ],
    correctAnswer: 1,
    explanation:
      'REALISTIC pricing model: Entry (buying): Pay ASK price + slippage + commission. Exit (selling): Receive BID price - slippage - commission. Or simplified: Use MID-PRICE ± slippage + commission. Example: Option mid = $5.00, bid-ask spread = $0.10, Buy: Pay $5.05 (mid + half spread) + $0.01 slippage + $0.65 commission = $5.71 total. Sell: Receive $4.95 (mid - half spread) - $0.01 slippage - $0.65 commission = $4.29. Transaction costs are ~3-5% of option price - very material to backtest results!',
  },
  {
    id: 'module-project-options-platform-mc-5',
    question:
      'For the Module Project deliverable, what is the minimum acceptable accuracy for Black-Scholes pricing compared to market prices?',
    options: [
      'Within $10 per contract',
      'Within 1-2% of the market price for liquid options',
      'Exact match to market price',
      'Within 50% of market price',
    ],
    correctAnswer: 1,
    explanation:
      'Acceptable Black-Scholes accuracy: Within 1-2% of market mid-price for LIQUID ATM options. Example: Market price $5.00 → BS should be $4.90-$5.10. Why not exact? Market prices reflect supply/demand, bid-ask spread, discrete strikes, early exercise (American options). BS assumes: Continuous trading, no bid-ask, European exercise, constant volatility. For ILLIQUID or far OTM: Tolerance can be 5-10% (wider spreads). Validation: Test on SPY/SPX options (most liquid) → should match within 1-2%. If error > 5%, likely implementation bug.',
  },
];
