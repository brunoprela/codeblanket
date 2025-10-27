export const portfolioGreeksManagementMC = [
  {
    id: 'portfolio-greeks-management-mc-1',
    question:
      'When managing a portfolio of options, what does "portfolio delta" represent?',
    options: [
      'The sum of all individual option prices',
      'The aggregate directional exposure, equivalent to number of shares long or short',
      'The total number of option contracts held',
      'The average time to expiration across all positions',
    ],
    correctAnswer: 1,
    explanation:
      'Portfolio Delta = Sum of all position deltas (considering quantity). Represents AGGREGATE DIRECTIONAL EXPOSURE equivalent to number of shares. Example: Long 10 calls (delta 0.50 each) = +500 delta, short 5 puts (delta -0.30 each) = +150 delta, total = +650 delta equivalent to long 650 shares. Target: Often near-zero (market-neutral) or small bias. Critical for risk management.',
  },
  {
    id: 'portfolio-greeks-management-mc-2',
    question:
      'What is the primary benefit of maintaining positive "portfolio gamma"?',
    options: [
      'Protection from time decay',
      'Profit from large stock moves through gamma scalping (rehedging)',
      'Guaranteed daily income',
      'Elimination of directional risk',
    ],
    correctAnswer: 1,
    explanation:
      'Positive Gamma benefits from LARGE MOVES through gamma scalping. As stock moves, delta changes → rehedge by buying low/selling high repeatedly. Gamma P&L = 0.5 × gamma × (stock_move)². Profitable when Realized Vol > Implied Vol. Example: Gamma +500, stock moves $5 → gamma P&L = 0.5 × 500 × 25 = $6,250. Requires active rehedging to capture. Not guaranteed profit (theta decay offsets).',
  },
  {
    id: 'portfolio-greeks-management-mc-3',
    question: 'For a portfolio with theta = +$2,000 per day, this indicates:',
    options: [
      'The portfolio loses $2,000 each day due to time decay',
      'The portfolio gains $2,000 each day from time decay (e.g., from selling options)',
      'The portfolio is at risk of $2,000 loss per day',
      'The portfolio has $2,000 in transaction costs daily',
    ],
    correctAnswer: 1,
    explanation:
      'POSITIVE Theta (+$2,000/day) means portfolio GAINS from time decay. Typical of: Short option strategies (collecting premium), iron condors, short strangles, covered calls. Portfolio benefits as time passes (options sold decay). Example: Sell 10 iron condors, collect $3 premium, theta +$200/day → after 30 days, collect ~$6,000 if stock stays in range. Negative theta means losing to decay (long options).',
  },
  {
    id: 'portfolio-greeks-management-mc-4',
    question:
      'In stress testing a portfolio, a "-20% stock move" scenario shows a loss of $800,000 on a $5M portfolio. What is the appropriate action?',
    options: [
      'No action needed, stress tests are just informational',
      'Immediately close all positions',
      'Reduce position sizes or add hedges since loss exceeds acceptable risk (>15%)',
      'Increase leverage to recover potential losses',
    ],
    correctAnswer: 2,
    explanation:
      'Loss of $800K = 16% of $5M portfolio, EXCEEDS typical risk limit (15% max). ACTION REQUIRED: Reduce delta exposure (buy/sell hedges), Add protective puts (insurance), Scale down position sizes, Close high-risk positions. Professional risk management sets limits BEFORE losses occur. Stress test purpose: Identify unacceptable risks proactively. Example: If target max loss is $750K (15%), current $800K → reduce positions.',
  },
  {
    id: 'portfolio-greeks-management-mc-5',
    question:
      'P&L attribution analysis shows: Delta +$5,000, Gamma +$2,000, Theta +$500, Vega -$8,000. What was the primary driver of the daily loss?',
    options: [
      'Directional move (delta)',
      'Large stock move (gamma)',
      'Time decay (theta)',
      'Implied volatility decrease (vega)',
    ],
    correctAnswer: 3,
    explanation:
      "Total P&L = $5K + $2K + $0.5K - $8K = -$0.5K (small loss). PRIMARY DRIVER: VEGA (-$8,000) from IV decrease. Portfolio was long vega (hurt by IV drop). Other Greeks were positive but couldn't offset vega loss. This shows: Portfolio is vega-sensitive (likely long straddles/options), Need to hedge vega exposure (add negative vega positions), Or reduce overall vega to lower sensitivity. Delta/gamma/theta all contributed positively, but vega dominated.",
  },
];
