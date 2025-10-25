import { MultipleChoiceQuestion } from '@/lib/types';

export const liquidityMarketImpactMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-13-mc-1',
    question:
      "Kyle\'s Lambda for a stock is 0.0001(trading 10,000 shares moves price $1).You need to buy 50,000 shares.Using the square- root model Impact = λ × Q ^ 0.5, what is the estimated market impact ? ",
    options: ['$1.00', '$2.24', '$5.00', '$7.07'],
    correctAnswer: 1,
    explanation:
      'Square-root model: Impact = λ × Q^0.5 = 0.0001 × 10,000 × √(50,000/10,000) = $1 × √5 = $1 × 2.236 = $2.24. This shows concavity: 5x size → only 2.24x impact (economies of scale). Linear model would predict $5. Real markets show ~0.5-0.6 exponent.',
  },
  {
    id: 'fm-1-13-mc-2',
    question:
      'Almgren-Chriss model: In high volatility environment (VIX = 40), should you execute a large order quickly or slowly?',
    options: [
      'Quickly (price risk dominates)',
      'Slowly (minimize market impact)',
      'No difference',
      "Don't trade at all",
    ],
    correctAnswer: 0,
    explanation:
      'High volatility → price risk dominates market impact cost. If vol = 3%/day and you trade over 5 days, price could move ±15% → huge risk. Better: Execute in 1 day, pay 0.5% impact, avoid 15% risk. Low vol → opposite: Trade slowly to minimize impact. Almgren-Chriss optimizes this trade-off.',
  },
  {
    id: 'fm-1-13-mc-3',
    question:
      'An order has 60% temporary market impact (recovers after 5 minutes) and 40% permanent impact (information-driven). After trading 10,000 shares with $1.00 total impact, how much recovers?',
    options: ['$0.40', '$0.60', '$1.00 (all recovers)', '$0 (none recovers)'],
    correctAnswer: 1,
    explanation:
      'Temporary impact = 60% × $1.00 = $0.60 recovers after execution completes. Permanent = 40% × $1.00 = $0.40 stays. This is why VWAP execution works: Spread over time, temporary impact recovers between trades, only pay permanent. Informed traders cause permanent impact (market learns from their trades).',
  },
  {
    id: 'fm-1-13-mc-4',
    question:
      'A dark pool has 80% fill rate but you suspect it leaks order information to HFTs. A transparent exchange has 100% fill but 0.15% higher cost. Which should you use?',
    options: [
      'Dark pool (80% fill rate is good)',
      'Exchange (transparency worth cost)',
      'Depends: Check if price moves after dark pool pings',
      'Use both equally',
    ],
    correctAnswer: 2,
    explanation:
      'Test for leakage: Ping dark pool, monitor if lit market price moves against you within milliseconds. If leakage detected: Dark pool "80% fill" comes with hidden 0.20%+ cost from front-running → worse than transparent exchange. If no leakage: Use dark pool. Always monitor for information leakage.',
  },
  {
    id: 'fm-1-13-mc-5',
    question:
      'Average Daily Volume (ADV) for a stock is $10M. You need to trade $1M (10% of ADV). This is considered:',
    options: [
      'Small order (<1% ADV)',
      'Medium order (1-5% ADV)',
      'Large order (5-20% ADV)',
      'Block trade (>20% ADV)',
    ],
    correctAnswer: 2,
    explanation:
      "Classification: <1% ADV = small (minimal impact), 1-5% = medium (noticeable impact), 5-20% = large (significant impact, need smart execution), >20% = block (negotiate off-market). At 10% ADV, expect 15-30 bps market impact. Need multi-hour VWAP or dark pool strategy. Don't dump at market!",
  },
];
