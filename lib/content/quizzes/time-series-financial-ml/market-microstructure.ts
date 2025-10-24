export const marketMicrostructureQuiz = [
  {
    id: 'mm-q-1',
    question: 'Explain bid-ask spread and how it affects trading strategies.',
    sampleAnswer:
      'Bid-ask spread: Difference between best buy price (bid) and best sell price (ask). Example: Bid $100.00, Ask $100.02 → spread $0.02 (0.02%). Cost: Every round-trip (buy+sell) costs spread. High-frequency: spread = major cost. Tight spread (0.01%) = liquid stocks, wide spread (0.5%+) = illiquid. Strategies: Market maker profits from spread, momentum traders pay spread. Reduce: Use limit orders, trade liquid stocks, avoid market orders.',
    keyPoints: [
      'Spread = Ask - Bid, cost of immediate execution',
      'Tight spread (0.01%) = liquid, wide (0.5%) = illiquid',
      'HFT: spread is major cost (100 trades/day × 0.02% = 2%)',
      'Reduce: limit orders, trade liquid stocks',
      'Market makers profit from spread',
    ],
  },
  {
    id: 'mm-q-2',
    question: 'Compare market orders vs limit orders. When to use each?',
    sampleAnswer:
      'Market order: Execute immediately at best price. Pros: Guaranteed fill, fast. Cons: Slippage (pay ask when buying). Limit order: Only execute at specified price. Pros: Price control, no slippage. Cons: May not fill, miss opportunities. Use market: Urgent, liquid stocks, small size. Use limit: Patient, illiquid stocks, large size. Best: Limit order at mid-price or slightly inside spread. Sophisticated: Use limit then convert to market if not filled in 1 minute.',
    keyPoints: [
      'Market: fast fill but pay spread + slippage',
      'Limit: price control but may not fill',
      'Market for: urgent, liquid, small orders',
      'Limit for: patient, illiquid, large orders',
      'Hybrid: limit order, cancel and use market if needed',
    ],
  },
  {
    id: 'mm-q-3',
    question: 'What is order flow imbalance? How do you use it for prediction?',
    sampleAnswer:
      'Order Flow Imbalance (OFI): (Buy volume - Sell volume) / Total volume. Positive OFI → more buyers, price likely up. Example: 1M shares bought, 800k sold → OFI = 0.11 (bullish). Use: (1) Trade in direction of OFI (short-term), (2) OFI > 0.2 → go long, (3) Predictive power 1-10 minutes. Combine with price action. Used by HFT firms. Limitation: Works only short-term (minutes), requires tick data. Profitable in liquid stocks.',
    keyPoints: [
      'OFI = (Buy vol - Sell vol) / Total vol',
      'Positive OFI → buying pressure → price up',
      'Predictive horizon: 1-10 minutes',
      'Use: trade in direction of OFI (>0.2 threshold)',
      'Requires tick data, works in liquid stocks',
    ],
  },
];
