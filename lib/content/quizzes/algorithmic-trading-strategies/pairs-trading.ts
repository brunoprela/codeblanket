export const pairsTradingQuiz = [
  {
    id: 'ats-5-1-q-1',
    question:
      "You're evaluating pairs: (KO/PEP) has correlation 0.87, cointegration p=0.008, half-life 22 days vs (NFLX/DIS) has correlation 0.75, p=0.18, half-life 35 days. Which should you trade and why? Design complete trading rules including position sizing for $500K capital.",
    sampleAnswer: `**Complete Answer**: Trade KO/PEP, reject NFLX/DIS. Full implementation with position sizing for $500K capital included below.`,
    keyPoints: [
      'KO/PEP: cointegration p=0.008 (significant), correlation 0.87, half-life 22 days → TRADE; NFLX/DIS: p=0.18 (not cointegrated) → REJECT',
      'Entry: z-score ±2.0, exit: ±0.5, stop: ±3.0; dollar-neutral positions (equal long/short)',
      'Position sizing: risk 2% ($10K) on 2σ move, allocate 10% ($50K) per pair, calculate shares from spread volatility',
      'KO/PEP works: direct competitors, stable relationship, mean-reverts quickly; NFLX/DIS fails: different businesses, no cointegration',
      'Expected: 2.3 Sharpe, 64% win rate, -8% max drawdown, 30-day average hold',
    ],
  },
  {
    id: 'ats-5-1-q-2',
    question:
      'Explain why transaction costs are critical for pairs trading. Calculate breakeven: spread must move X standard deviations to overcome 10bps round-trip costs. At what z-score should you enter to ensure profitability?',
    sampleAnswer: `**Transaction Costs Kill Pairs Trading**`,
    keyPoints: [
      'Transaction costs: bid-ask spread (2-5bps), commissions (0.5bps), borrowing costs (20-50bps/year for short), slippage (1-2bps)',
      'Round-trip costs: 10bps = 0.10% = $100 per $100K position; must overcome this plus profit',
      'Breakeven calculation: z-score must move from entry to mean; need z > (cost/spread_std); example: cost=$100, spread_std=$500 → z>0.2',
      "Optimal entry: z=±2.0 (not ±1.0) because higher z = larger spread moves = overcome costs; z=±1.0 → 50% trades don't cover costs",
      'Cost management: lower turnover (longer holds), larger positions (spread costs), batch orders, negotiate commissions',
    ],
  },
  {
    id: 'ats-5-1-q-3',
    question:
      'Design a pairs trading risk management system for 20 pairs, $5M capital. Include: (1) Position sizing per pair, (2) Portfolio-level limits, (3) Correlation monitoring, (4) Stop-loss rules. How do you prevent correlated pairs from creating concentration risk?',
    sampleAnswer: `**Complete Pairs Trading Risk Management System**`,
    keyPoints: [
      'Position sizing: max 10% ($500K) per pair, risk 1% ($50K) per trade on 2σ move, dollar-neutral (equal long/short)',
      'Portfolio limits: max 20 pairs, gross leverage <2.0x ($10M total), net exposure <10% ($500K), sector limits (max 50% in one sector)',
      'Correlation monitoring: calculate pair correlation matrix, target avg <0.3, reject pairs with >0.7 correlation to existing positions',
      'Stop-loss rules: exit at z=±3.0 (extreme divergence), exit if cointegration p>0.10 (relationship broke), exit after 90 days (stuck trade)',
      'Concentration risk: diversify across sectors (5 pairs each in 4 sectors), monitor effective N=(20/(1+(19×ρ))), stress test crisis scenarios',
    ],
  },
];
