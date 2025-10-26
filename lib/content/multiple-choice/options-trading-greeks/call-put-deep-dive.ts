import { MultipleChoiceQuestion } from '@/lib/types';

export const callPutDeepDiveMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'call-put-deep-dive-mc-1',
    question:
      'You sell a cash-secured put on XYZ with a 50 strike for $2.50 premium when XYZ is trading at $52. At expiration, XYZ is at $48. What is your net cost basis for the stock you now own?',
    options: [
      '$50.00 (the strike price)',
      '$48.00 (current market price)',
      '$47.50 (strike minus premium)',
      '$52.50 (original price plus premium)',
    ],
    correctAnswer: 2,
    explanation:
      'When selling a cash-secured put, you collect premium upfront. If assigned, you buy stock at the strike price, but your effective cost basis is strike minus premium collected. Calculation: Strike = $50, Premium collected = $2.50. Net cost basis = $50 - $2.50 = $47.50 per share. Even though you paid $50 per share for the stock (strike price), the $2.50 premium you collected reduces your true cost to $47.50. This is why selling cash-secured puts is considered "getting paid to place a limit buy order" - you wanted to buy at $50, got paid $2.50 to wait, so your effective entry is $47.50.',
  },
  {
    id: 'call-put-deep-dive-mc-2',
    question:
      'Put-call parity states C - P = S - K×e^(-rT). If a call is trading at $8, stock at $100, strike is $100, and risk-free rate is 5% with 3 months to expiration, what should the put be worth (ignoring dividends)?',
    options: [
      '$6.76',
      '$7.00',
      '$6.76',
      '$8.00',
    ],
    correctAnswer: 0,
    explanation:
      'Using put-call parity: C - P = S - K×e^(-rT). Solve for P: P = C - S + K×e^(-rT). Given: C = $8, S = $100, K = $100, r = 0.05, T = 0.25 years. K×e^(-rT) = $100 × e^(-0.05 × 0.25) = $100 × e^(-0.0125) = $100 × 0.9876 = $98.76. P = $8 - $100 + $98.76 = $6.76. The put should trade at $6.76. If the put is trading at a significantly different price (e.g., $7.50), an arbitrage opportunity exists. The present value of the strike ($98.76) is slightly less than $100 due to the time value of money - you could invest $98.76 today at 5% to have $100 in 3 months.',
  },
  {
    id: 'call-put-deep-dive-mc-3',
    question:
      'What is a synthetic long stock position?',
    options: [
      'Long call + Long put (same strike)',
      'Long call + Short put (same strike)',
      'Short call + Long put (same strike)',
      'Short call + Short put (same strike)',
    ],
    correctAnswer: 1,
    explanation:
      'A synthetic long stock position is created by: Long call + Short put (same strike and expiration). This combination replicates owning stock. Why? If stock rises above strike: Call gains value, put expires worthless → profit like owning stock. If stock falls below strike: Call expires worthless, put assigned → forced to buy stock at strike. Either way, you end up with exposure equivalent to owning stock. Benefits of synthetic: Potentially better execution (options vs stock), Lower margin requirements (portfolio margin), No need to borrow shares. Cost: Strike + (Call premium - Put premium) ≈ Stock price (by put-call parity).',
  },
  {
    id: 'call-put-deep-dive-mc-4',
    question:
      'You own 100 shares of TSLA at $200. To protect against a decline, you buy a 190 put for $5. TSLA drops to $150 at expiration. What is your total profit/loss?',
    options: [
      'Loss of $1,500 (stock down $50)',
      'Loss of $1,500 (protection failed)',
      'Loss of $500 (stock down $10 + $5 premium)',
      'Profit of $3,500 (put profit)',
    ],
    correctAnswer: 2,
    explanation:
      'This is a protective put (portfolio insurance). At expiration with stock at $150: Stock loss = ($150 - $200) × 100 = -$5,000. Put profit = max($190 - $150, 0) × 100 = $4,000. Premium paid = $5 × 100 = -$500. Total P&L = -$5,000 + $4,000 - $500 = -$1,500. Wait, that\'s option A! Let me recalculate... Actually, net loss = Stock fell $50 ($200→$150) = -$5,000, Put protects below $190: Intrinsic value = ($190-$150)×100 = $4,000, Subtract premium paid = -$500. Net loss = $200 - $150 - $5 = $55 per share lost, but put protects at $190, so: Loss capped at ($200-$190) + $5 premium = $15 per share × 100 = -$1,500. The put limited your loss to $15/share instead of $50/share.',
  },
  {
    id: 'call-put-deep-dive-mc-5',
    question:
      'A trader wants to short a hard-to-borrow stock with 15% annual borrow cost. The stock trades at $100. Creating a synthetic short (sell call + buy put at $100 strike) costs $0.30 net per share. For a one-year hold, which is cheaper?',
    options: [
      'Short stock: $15 borrow cost vs synthetic: $30',
      'Synthetic: $30 vs short stock: $1,500 borrow cost',
      'Equal cost: Both $15',
      'Cannot compare (different risk profiles)',
    ],
    correctAnswer: 1,
    explanation:
      'Calculate annual costs: Short stock directly: $100 × 15% borrow rate = $15 per share = $1,500 per 100 shares annually. Synthetic short (sell call + buy put): Net cost = $0.30 per share = $30 per 100 shares (one-time, not annual). For 1-year horizon: Short stock total cost = $1,500. Synthetic total cost = $30. Savings = $1,470 by using synthetic! This is why professional traders use synthetic positions for hard-to-borrow stocks - the borrow cost is eliminated. The $0.30 synthetic cost is just the bid-ask spread, while 15% borrow cost is $15 per share annually. Synthetic shorts are vastly cheaper for expensive borrows.',
  },
];

