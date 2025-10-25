import { MultipleChoiceQuestion } from '@/lib/types';

export const tradingVenuesExchangesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-10-mc-1',
    question:
      'An institution needs to execute a $10M buy order. Average Daily Volume (ADV) is $100M. Using the market impact model Impact = 10 × (Size/ADV)^0.6, what is the estimated market impact in basis points?',
    options: ['10 bps', '15 bps', '25 bps', '40 bps'],
    correctAnswer: 2,
    explanation:
      'Impact = 10 × (10M / 100M)^0.6 = 10 × (0.1)^0.6 = 10 × 0.251 = 2.51 ≈ 25 bps. This is why large orders need smart routing - dumping $10M at once costs 0.25%! Using dark pools and algos can reduce this to 10-15 bps.',
  },
  {
    id: 'fm-1-10-mc-2',
    question:
      'An order shows Implementation Shortfall of 12 bps, market impact of 8 bps, timing cost of -3 bps (favorable), and opportunity cost of 3 bps. What is the total transaction cost and execution grade?',
    options: [
      '12 bps (Excellent)',
      '20 bps (Good)',
      '23 bps (Fair)',
      '26 bps (Poor)',
    ],
    correctAnswer: 1,
    explanation:
      'Total cost = IS + Impact + Timing + Opportunity = 12 + 8 + (-3) + 3 = 20 bps. Grading: <5 bps = Excellent, <15 = Good, <30 = Fair, >30 = Poor. 20 bps = Good execution (B grade). Under 20 bps for large orders is solid performance.',
  },
  {
    id: 'fm-1-10-mc-3',
    question:
      'Co-location at an exchange costs $50,000/month and reduces latency from 5 milliseconds to 50 microseconds. What is the latency advantage?',
    options: ['10x faster', '50x faster', '100x faster', '1000x faster'],
    correctAnswer: 2,
    explanation:
      '5 milliseconds = 5,000 microseconds. Advantage = 5,000µs / 50µs = 100x faster. This allows HFT firms to see and react to orders 100x faster than non-colocated participants. Annual cost: $50K × 12 = $600K. Firms spend $100M+ on speed infrastructure for this advantage.',
  },
  {
    id: 'fm-1-10-mc-4',
    question:
      'IEX Exchange implements a 350-microsecond speed bump for all orders. A co-located HFT firm has 50µs latency, while a retail investor has 5,000µs. After the speed bump, what is the relative advantage?',
    options: [
      'Still 100x faster (5,000µs vs 50µs)',
      'Reduced to 12.5x faster (5,350µs vs 400µs)',
      'Eliminated within exchange (both 350µs)',
      'Retail is now faster',
    ],
    correctAnswer: 2,
    explanation:
      'Speed bump: 350µs added to ALL orders equally. Within the exchange matching engine, all orders arrive with same 350µs delay → eliminates latency arbitrage. HFT still faster getting TO exchange, but once there, everyone equal. This is why IEX model works - preserves liquidity while ending speed wars.',
  },
  {
    id: 'fm-1-10-mc-5',
    question:
      'Dark pools provide what key advantage for large institutional orders?',
    options: [
      'Lower fees than lit exchanges',
      'Faster execution speed',
      'Orders are hidden (no pre-trade transparency)',
      'Access to retail flow',
    ],
    correctAnswer: 2,
    explanation:
      'Dark pools: No pre-trade transparency → orders hidden until executed. This prevents front-running of large orders. If institution shows "Buy 100K shares" on lit exchange, HFTs buy ahead, pushing price up. Dark pools: Find natural liquidity (other institutions), minimal market impact. Trade-off: Lower fill rates than lit markets.',
  },
];
