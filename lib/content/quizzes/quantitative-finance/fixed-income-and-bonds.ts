export const fixedIncomeAndBondsQuiz = {
  id: 'fixed-income-and-bonds-quiz',
  title: 'Fixed Income & Bonds - Discussion Questions',
  questions: [
    {
      id: 'fib-q1',
      question:
        'A pension fund has a liability of $500M due in 15 years. The fund manager proposes two strategies: Strategy A: Buy $500M face value of 15-year zero-coupon Treasury bonds currently priced at $400M (YTM 4.2%). Strategy B: Buy a portfolio of 10-year (duration 8.5) and 20-year (duration 14.5) coupon bonds weighted to achieve 15-year duration, total investment $420M. Current rates: 10Y at 4.0%, 20Y at 4.5%. (1) Calculate the exact portfolio weights for Strategy B to match 15-year duration, (2) Compare interest rate risk if rates rise 1% uniformly (parallel shift), (3) Analyze convexity differences and impact on hedging effectiveness, (4) Evaluate reinvestment risk for each strategy, (5) Recommend optimal strategy considering transaction costs, rebalancing needs, and basis risk.',
      sampleAnswer: `Answer to be completed.`,
      keyPoints: [
        'Strategy B weights: 31.6% 10Y, 68.4% 30Y to achieve 15Y duration; cannot use only 10Y+20Y (requires shorting)',
        'Rate rise 1%: Both lose 14.39% (duration-matched), but B loses more dollars ($60.4M vs $57.6M) due to higher initial cost',
        'Strategy B has higher convexity (230.5 vs 216) → better protection for large rate moves (+0.07% benefit)',
        'Reinvestment risk: Strategy A has ZERO (critical advantage), Strategy B has HIGH (coupons + early maturity → $40M shortfall risk)',
        'Recommendation: Strategy A (zero-coupon) - saves $20M upfront + $2.6M transaction costs, no rebalancing, perfect liability match',
      ],
    },
    {
      id: 'fib-q2',
      question:
        'You observe the following Treasury yields: 1Y: 3.5%, 2Y: 4.0%, 3Y: 4.3%, 5Y: 4.6%, 10Y: 4.8%, 30Y: 5.0%. The yield curve is upward sloping but flattening (10Y-2Y spread = 80 bps, historically 10Y-2Y averages 150 bps). You are considering three fixed income strategies: (1) Carry Trade: Borrow 2Y at 4.0%, invest 10Y at 4.8% (positive carry 80 bps), (2) Curve Steepener: Long 10Y bonds, short 2Y bonds (duration-neutral), betting curve steepens back to 150 bps, (3) Bullet Strategy: Buy 5Y bonds, hold to maturity. For each strategy: (A) Calculate expected return/PnL assuming: (i) Rates unchanged, (ii) Parallel shift up 1%, (iii) Curve steepens (2Y +0.5%, 10Y unchanged), (B) Analyze breakeven scenarios, (C) Identify key risks, (D) Recommend optimal strategy given Fed is expected to cut rates in 12 months (priced in market already).',
      sampleAnswer: `Answer to be completed.`,
      keyPoints: [
        'Carry trade: +1.2% if unchanged, -5.8% if rates +1% (12 bps breakeven); high risk from duration mismatch (net D=6.6)',
        'Curve steepener: -16.9% if unchanged (negative carry -13.1%), needs 154 bps steepening to break even (unrealistic)',
        'Bullet 5Y: +5.3% if unchanged, +0.1% if rates +1% (102 bps breakeven); best risk/reward, protected by carry',
        'Steepener has 4.5× leverage ($447M short for $100M long) with massive negative carry; expensive bet',
        'Recommendation: Bullet 5Y - balanced carry (4.6%), wide breakeven, works if Fed cuts priced-in or not realized',
      ],
    },
    {
      id: 'fib-q3',
      question:
        'A corporate bond portfolio manager holds $50M in BBB-rated 5-year bonds from Company XYZ, currently trading at par (100) with 5% coupon, YTM 5.0%, duration 4.3. Credit spreads: BBB = 150 bps over Treasuries (5Y Treasury at 3.5%). You receive credit research suggesting XYZ may be downgraded to BB (high yield) in 6 months due to deteriorating fundamentals. Historical data: BBB→BB downgrades cause spread widening from 150 bps to 400 bps (average), and price impacts of -10% to -15%. You consider three risk management approaches: (1) Sell immediately at current price (100), lock in par value, (2) Buy 5Y credit default swap (CDS) protection at 180 bps annually, (3) Sell 30% of position now, buy CDS on remaining 70%. (A) Calculate expected PnL for each strategy under scenarios: (i) No downgrade (spreads tighten to 120 bps), (ii) Downgrade occurs (spreads widen to 400 bps), (iii) Default occurs (recovery rate 40%). (B) Determine breakeven CDS cost, (C) Analyze optimal hedge ratio, (D) Recommend strategy considering: (i) 60% probability of downgrade, (ii) 5% probability of default, (iii) Transaction costs 0.2% for bond sales, (iv) Opportunity cost if wrong (spreads tighten).',
      sampleAnswer: `Answer to be completed.`,
      keyPoints: [
        'Strategy comparison: Sell all (+$5.0M expected), CDS 100% (-$1.6M), Sell 30%/CDS 70% (-$1.1M), Hold (-$5.0M)',
        'CDS inefficient: Only protects default (5% prob), not downgrade (60% prob); premium 180 bps too expensive',
        'Breakeven: CDS worth buying if default prob > 3% (actual 5%), but downgrade scenario (60%) makes CDS negative EV',
        'Optimal hedge ratio: 100% (sell all) maximizes expected value; (1-h) × (-$5M) minimized when h=1',
        'Recommendation: Sell immediately - save $6.25M downgrade loss vs $645k opportunity cost (risk-reward 10:1)',
      ],
    },
  ],
};
