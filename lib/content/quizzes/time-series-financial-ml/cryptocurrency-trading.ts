export const cryptocurrencyTradingQuiz = [
  {
    id: 'ct-q-1',
    question:
      'What are the key differences between crypto and stock trading? How do you adapt strategies?',
    sampleAnswer:
      'Differences: (1) 24/7 trading (no close, weekend gaps), (2) High volatility (2-5x stocks), (3) Unique data (on-chain, funding rates), (4) Lower liquidity (large slippage), (5) More manipulation. Adaptations: (1) Tighter stops (2-3% vs 5%), (2) Smaller positions (1% vs 2% risk), (3) Shorter holding (hours vs days), (4) Use on-chain data (exchange flows, active addresses), (5) Higher Sharpe potential (1.5-2.0 achievable). More opportunities but more risk. Need 24/7 monitoring or automated system.',
    keyPoints: [
      '24/7 trading, 2-5x volatility, unique data sources',
      'Tighter stops (2-3%), smaller positions (1% risk)',
      'Shorter holding periods (hours vs days)',
      'Use on-chain metrics, funding rates, sentiment',
      'Higher risk but higher Sharpe potential (1.5-2.0)',
    ],
  },
  {
    id: 'ct-q-2',
    question: 'What are on-chain metrics? How do you use them for trading?',
    sampleAnswer:
      'On-chain: Data from blockchain (transactions, addresses, flows). Key metrics: (1) Exchange inflow: BTC moving to exchanges (bearish—selling pressure), (2) Exchange outflow: BTC leaving exchanges (bullish—hodling), (3) Active addresses: Network activity, (4) Transaction volume: On-chain activity, (5) MVRV ratio: Market value / Realized value (>3.7 = overheated). Use: Exchange net flow > 10k BTC/day → bearish signal. Active addresses declining → bearish. Combine with price. Sources: Glassnode, CryptoQuant. Unique edge over technical analysis.',
    keyPoints: [
      'On-chain: blockchain transaction data',
      'Exchange inflow (bearish), outflow (bullish)',
      'Active addresses, transaction volume, MVRV',
      'Net flow > threshold → trading signal',
      'Sources: Glassnode, CryptoQuant, unique edge',
    ],
  },
  {
    id: 'ct-q-3',
    question:
      'Design risk management for crypto trading. Why is it more important than stocks?',
    sampleAnswer:
      'Crypto needs stricter risk: (1) Position: 1% risk per trade (vs 2% stocks), (2) Stop-loss: 2-3% (vs 5% stocks), (3) Max positions: 5 (vs 10), (4) Total exposure: 50% max (rest in stablecoins), (5) Daily loss limit: 3% (vs 5%), (6) No leverage initially (20x leverage = instant liquidation). Why stricter: (1) 40-60% annual volatility (vs 15-20% stocks), (2) Flash crashes common (-20% in hours), (3) Exchange hacks/downtime, (4) 24/7 = hard to monitor. One bad day can wipe account. Paranoid risk management essential. Many blown accounts from overleveraging.',
    keyPoints: [
      'Tighter: 1% risk/trade, 2-3% stops, 5 max positions',
      'Total exposure 50% max, rest stablecoins',
      'Daily loss limit 3%, no/low leverage',
      'Why: 40-60% vol, flash crashes, 24/7',
      'Overleveraging #1 cause of blown accounts',
    ],
  },
];
