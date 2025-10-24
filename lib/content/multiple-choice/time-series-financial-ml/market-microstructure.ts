import { MultipleChoiceQuestion } from '@/lib/types';

export const marketMicrostructureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mm-mc-1',
    question: 'What is bid-ask spread?',
    options: [
      'Difference between yesterday and today price',
      'Difference between best buy price and best sell price',
      'Stock volatility',
      'Trading volume',
    ],
    correctAnswer: 1,
    explanation:
      'Bid-ask spread: Best buy price (bid) vs best sell price (ask). Example: Bid $100.00, Ask $100.02 → spread $0.02. Cost of immediate execution. Tight spread (0.01%) = liquid, wide (1%+) = illiquid. Every trade pays half spread.',
  },
  {
    id: 'mm-mc-2',
    question: 'What is a market order?',
    options: [
      'Order executed at specified price',
      'Order executed immediately at best available price',
      'Order never executed',
      'Cancelled order',
    ],
    correctAnswer: 1,
    explanation:
      'Market order: Execute immediately at best price. Buy → pay ask price, Sell → receive bid price. Pros: Guaranteed fill, fast. Cons: Slippage (unfavorable price), especially in illiquid stocks. Use for urgent trades or small size.',
  },
  {
    id: 'mm-mc-3',
    question: 'What is a limit order?',
    options: [
      'Order with no price limit',
      'Order executed only at specified price or better',
      'Random order',
      'Market order',
    ],
    correctAnswer: 1,
    explanation:
      'Limit order: Execute only at limit price or better. Buy limit $100 → only buy if price ≤ $100. Pros: Price control, no slippage. Cons: May not fill if price moves away. Use for patient traders, large orders, or illiquid stocks.',
  },
  {
    id: 'mm-mc-4',
    question: 'What is VWAP?',
    options: [
      'Very Wide Asset Price',
      'Volume-Weighted Average Price',
      'Volatility-Weighted Average',
      'Random price',
    ],
    correctAnswer: 1,
    explanation:
      'VWAP: Volume-weighted average price. Weighted average of all trades during period. Benchmark for execution quality. If buy below VWAP → good execution. Large traders aim to match VWAP. Used by institutions to measure trading desk performance.',
  },
  {
    id: 'mm-mc-5',
    question: 'What is order flow imbalance (OFI)?',
    options: [
      'Random metric',
      '(Buy volume - Sell volume) / Total volume',
      'Price change',
      'Bid-ask spread',
    ],
    correctAnswer: 1,
    explanation:
      'OFI: (Buy vol - Sell vol) / Total vol. Measures buying vs selling pressure. OFI > 0 → more buyers (bullish), OFI < 0 → more sellers (bearish). Predictive for 1-10 minutes. Used by HFT traders. Requires tick-level data.',
  },
];
