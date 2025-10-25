export const optionsPricingMentalMathQuiz = [
  {
    id: 'opmm-q-1',
    question:
      'You are in an Optiver interview and given: "Stock trading at $200, 30-day ATM call is $8. Without a calculator, estimate: (1) implied volatility, (2) the delta, (3) daily theta, (4) what the call price would be if volatility increased to your calculated level + 5%." Walk through your complete mental math process, showing all approximations and shortcuts. How accurate would you expect your estimates to be?',
    sampleAnswer:
      'Complete mental math solution: (1) Implied volatility: Use ATM formula C ≈ 0.4 × S × σ × √T. Here: 8 ≈ 0.4 × 200 × σ × √(30/365). First, √(30/365) = √(1/12.17) ≈ √(1/12) ≈ 0.29. So: 8 ≈ 0.4 × 200 × σ × 0.29 = 23.2σ. Therefore σ ≈ 8/23.2 ≈ 0.345 = 34.5%. (2) Delta: ATM option has delta ≈ 0.5. (3) Daily theta: Θ ≈ -0.5 × S × σ / √T per year. Θ ≈ -0.5 × 200 × 0.345 / 0.29 ≈ -34.5 / 0.29 ≈ -$119 per year. Per day: -119/365 ≈ -$0.33 per day. (4) New vol = 34.5% + 5% = 39.5%. New call price: C ≈ 0.4 × 200 × 0.395 × 0.29 ≈ 0.4 × 22.91 ≈ $9.16. Verification: original was $8, we added 5% vol. Vega for this option ≈ 0.004 × 200 × 0.29 ≈ $0.23 per 1% vol change. So 5% increase → $0.23 × 5 = $1.15 increase → $8 + $1.15 = $9.15. Matches! Accuracy: For ATM options with reasonable time to expiration, these approximations are within 2-3% of true values. The key sources of error: (1) approximating √(30/365) as √(1/12), (2) using 0.4 constant instead of exact normal distribution values, (3) ignoring interest rates (but justified for 30 days). Interview communication: state each step, show verification with alternative method (vega check), acknowledge approximation errors, demonstrate confidence in the approach.',
    keyPoints: [
      'Implied vol: σ ≈ C / (0.4 × S × √T) = 8 / (0.4 × 200 × 0.29) ≈ 34.5%',
      'ATM delta ≈ 0.5 by definition',
      'Daily theta ≈ -0.5 × S × σ / √T / 365 ≈ -$0.33 per day',
      'Price change from vol: use vega ≈ 0.004 × S × √T per 1% vol',
      'Mental math accurate within 2-3% for ATM options',
    ],
  },
  {
    id: 'opmm-q-2',
    question:
      'IMC trading interview: "A stock is at $100. You observe: 3-month $100 call trading at $5.50, 3-month $100 put trading at $4.50. Assuming risk-free rate is effectively zero, what do you do and why?" Then: "After your trade, the stock moves to $105. You had delta-hedged initially. Explain what happens to your position and how much profit you would make on $1M notional." Provide complete analysis including Greek exposures and P&L calculation.',
    sampleAnswer:
      'Complete arbitrage analysis: (1) Put-call parity check: C - P should equal S - K (with r=0). Observed: C - P = $5.50 - $4.50 = $1.00. Expected: S - K = $100 - $100 = $0. Violation: Call is overpriced by $1! (2) Arbitrage trade: Sell call (+$5.50), Buy put (-$4.50), Buy stock (-$100). Net: -$99, which at expiration is worth exactly -$100 (K strike), regardless of stock price. Profit: $1 per share, risk-free. (3) Alternative: Sell call, buy put (creates synthetic short stock at $100). Hold until expiration or delta-hedge. (4) For $1M notional: $1M / $100 = 10,000 shares. Profit = 10,000 × $1 = $10,000 risk-free. (5) Stock moves to $105 with delta hedge: Initially, we sold call (delta -0.5) and bought put (delta -0.5) for combined delta -1.0 (equivalent to short stock). Delta hedge: buy 1 share per contract to neutralize. When stock moves $5: our sold call loses $5 × 0.5 = $2.50, our put loses $5 × 0.5 = $2.50, total loss $5 from options. But our hedge (long 1 share) gains $5. Net: $0 from delta-hedged position PLUS we locked in $1 arbitrage profit. However, gamma exposure: as stock moves, deltas change. Sold call gamma ≈ -0.027 (from earlier), put gamma ≈ +0.027 (long put), net gamma ≈ 0. So minimal rehedging needed. P&L: $10,000 arbitrage profit on $1M notional is locked in regardless of stock movement if properly hedged. Interview communication: identify violation immediately, state arbitrage trade, explain P&L is locked in, acknowledge Greeks for hedging.',
    keyPoints: [
      'Put-call parity: C - P = $1, but S - K = $0, so $1 arbitrage',
      'Trade: sell call, buy put, buy stock → lock in $1 per share',
      'On $1M notional (10,000 shares): $10,000 risk-free profit',
      'Delta hedge neutralizes directional risk from stock movement',
      'Gamma-neutral (long put gamma offsets short call gamma)',
    ],
  },
  {
    id: 'opmm-q-3',
    question:
      'Jane Street interview: "You\'re a market maker in SPY options. A client wants to buy 1,000 contracts of 1-week 0.5% OTM calls. SPY is at $400, so strike is $402. Current implied vol is 15% for ATM, but you expect vol to increase to 20% this week due to upcoming Fed announcement. How would you price these options to the client, and how would you hedge your exposure? Include: (1) option price calculation at both vol levels, (2) Greek exposures, (3) hedging strategy including frequency, (4) risk management considerations." Provide comprehensive analysis suitable for a senior trader role.',
    sampleAnswer:
      'Comprehensive market-making analysis: (1) Option pricing: At current 15% vol: ATM price ≈ 0.4 × 400 × 0.15 × √(7/365) = 0.4 × 400 × 0.15 × 0.139 ≈ $3.34. For 0.5% OTM, adjust: moneyness factor ≈ √(400/402) ≈ 0.9975 ≈ 1 (minimal adjustment for 0.5%). OTM price ≈ $3.34 × 0.9975 ≈ $3.33. At expected 20% vol: $3.33 × (20/15) ≈ $4.44. So if I sell at 15% vol ($3.33) and vol goes to 20%, I lose ~$1.11 per contract. (2) Pricing to client: Add buffer for vol risk. Sell at implied vol of 17-18% (midpoint) to protect against vol increase. At 17.5% vol: price ≈ $3.33 × (17.5/15) ≈ $3.89. Offer to client: $3.90-4.00, marking up for vol risk and transaction costs. (3) Greek exposures for 1,000 contracts: Delta: 0.5% OTM has delta ≈ 0.48. Short 1,000 calls → delta -480. Gamma: Γ ≈ 0.4 / (400 × 0.15 × 0.139) ≈ 0.048. Short gamma ≈ -48. Vega: ν ≈ 0.004 × 400 × 0.139 ≈ $0.22 per 1% vol per contract. Short vega = -$220 per 1% vol change. Theta: +decay from short position. (4) Hedging strategy: Initial: Buy 480 SPY shares to delta-hedge at ~$400 = $192,000 capital. Gamma hedging: Short gamma means as stock rises, we need to buy more shares (buy high); as it falls, sell (sell low). This is expensive with large gamma. Consider buying some OTM puts for gamma protection. Rehedge frequency: With 1-week expiry and high gamma, rehedge every $1-2 move in SPY or at least daily. Vega hedge: Long vega (-220 short) by buying longer-dated ATM options. This protects against vol increase to 20%. (5) Risk management: Max loss scenario: vol jumps to 25%, stock moves to $410 (deep ITM). Calculate worst case: option value at 25% vol, $410 stock ≈ intrinsic $8 + time value. Mark-to-market daily, set stop-loss at $0.50 per contract loss = $500 total. Consider dynamic hedging costs (~$0.10 per contract expected from gamma rehedging). Final client price: $4.00 per contract to cover vol risk, hedging costs, and profit margin. Interview communication: demonstrate understanding of vol surface, Greek risk management, capital requirements for hedging, realistic trading costs, and risk limits.',
    keyPoints: [
      'Option price sensitive to vol: 15% → $3.33, 20% → $4.44 per contract',
      'Mark up to 17-18% vol ($3.90-4.00) to protect against expected vol increase',
      'Short 1,000 calls: delta -480, gamma -48, vega -$220 per 1% vol',
      'Hedge: buy 480 shares initially, rehedge daily or on $1-2 moves',
      'Vega hedge with longer-dated options, set risk limits, include hedging costs',
    ],
  },
];
