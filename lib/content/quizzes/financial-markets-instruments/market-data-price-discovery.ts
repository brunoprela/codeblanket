export const marketDataPriceDiscoveryQuiz = [
  {
    id: 'fm-1-12-q-1',
    question:
      'Design a real-time order book visualization and analysis system. Include: depth chart display, order book imbalance calculation, large order detection, and price level clustering. How do you detect spoofing (fake large orders that get cancelled)?',
    sampleAnswer: `[Implementation showing order book data structure, Level 2 market data processing, imbalance ratio calculation, and spoofing detection with high cancel-rate alerts]`,
    keyPoints: [
      'Order book: Real-time bid/ask prices with size at each level (Level 2 data)',
      'Imbalance: Bid size / Ask size. >2.0 suggests buying pressure, <0.5 selling pressure',
      'Large orders: Flag orders >2% of daily volume (might be institutional or spoof)',
      'Spoofing detection: Track cancel rate. If order >1000 shares cancelled >90% of time = spoof',
      'Price discovery: Watch for level changes, large sweeps, hidden liquidity revelations',
    ],
  },
  {
    id: 'fm-1-12-q-2',
    question:
      'Market data vendors (Bloomberg, Refinitiv) charge $2K+/month for real-time feeds. Design a cost-effective alternative for retail traders using: free delayed quotes, websocket feeds, and exchange APIs. What latency/feature trade-offs exist?',
    sampleAnswer: `[Architecture using Polygon.io free API, exchange websockets, 15-minute delayed data for non-urgent needs, and latency management strategies]`,
    keyPoints: [
      'Free options: Exchange APIs (15-min delay), Polygon.io (real-time for $200/mo), IEX Cloud',
      'Latency: Professional feeds 1-10ms, retail websockets 100-500ms (acceptable for non-HFT)',
      'Features: Bloomberg has news/analytics, free feeds just prices (99% of retail needs)',
      'DIY stack: Polygon + PostgreSQL + Grafana = $200/mo vs $2K Bloomberg',
      'Trade-off: No Bloomberg terminal features, but price data is 95% of value for most',
    ],
  },
  {
    id: 'fm-1-12-q-3',
    question:
      'Bid-ask spread reflects liquidity and information asymmetry. Build a real-time spread analysis system that: tracks spread over time, compares to historical norms, alerts on unusual widening, and estimates transaction costs.',
    sampleAnswer: `[System monitoring tick-by-tick spreads, calculating percentile bands, detecting 2-sigma widenings, and correlating with volatility and volume]`,
    keyPoints: [
      'Normal spread: Liquid stocks 1-5 bps ($0.01 on $100 stock). Illiquid: 50+ bps',
      'Spread widening signals: Low liquidity, high volatility, informed trading, or news pending',
      'Alert: If spread >2× 30-day average, investigate (might be earnings, halt imminent)',
      'Transaction cost: Spread/2 = one-way cost. $0.10 spread = 5 bps per trade (10 bps round-trip)',
      'Pattern: Spreads widen at open/close, tighten mid-day (monitor for deviations)',
    ],
  },
];
