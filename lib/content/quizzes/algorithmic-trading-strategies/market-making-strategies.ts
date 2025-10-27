export const marketMakingStrategiesQuiz = [
  {
    id: 'ats-7-1-q-1',
    question:
      'Design market making strategy for liquid stock: (1) Spread calculation, (2) Inventory management, (3) Adverse selection protection. Calculate expected daily P&L on 100K shares at 5bp spread.',
    sampleAnswer: `**Market Making Strategy Design**: (1) Spread: 5bps (0.05%), dynamically widen to 10bps when inventory >50%; (2) Inventory: max ±1000 shares, skew quotes when inventory builds; (3) Adverse selection: cancel orders within 100ms of news, widen spread during volatility spikes. Expected P&L: 100K shares × 5bps = $500/day gross, minus $100 adverse selection, $50 infrastructure = $350/day net.`,
    keyPoints: [
      'Spread management: base 5bps, widen to 10bps at inventory limits, tighten to 3bps in low vol',
      'Inventory risk: skew quotes (move both bid/ask) to push inventory back to zero; max position ±1000 shares',
      'Adverse selection: informed traders pick off stale quotes; protection via fast cancels (<100ms), spread widening, volume analysis',
      'P&L calculation: 100K shares/day × 5bps = $500 gross; costs: adverse selection $100, infrastructure $50; net $350/day',
      'Risk management: inventory limits, spread widening in volatility, hedging with futures when inventory >500 shares',
    ],
  },
  {
    id: 'ats-7-1-q-2',
    question:
      'Explain adverse selection in market making. How do high-frequency market makers protect against being picked off by informed traders?',
    sampleAnswer: `**Adverse Selection**: Informed traders trade against market makers before price moves (market maker loses). Protection: (1) Ultra-fast order cancellation (<100μs), (2) Predictive models detect informed flow, (3) Widen spreads during news/volatility, (4) Co-location (minimize latency), (5) Queue position monitoring. HFT market makers update quotes 1000+ times/second to avoid stale prices.`,
    keyPoints: [
      'Adverse selection: informed traders know price will move, trade against stale market maker quotes before MM can cancel',
      'Cost: adverse selection can eat 2-3bps of 5bp spread, reducing profitability by 40-60%',
      'Protection mechanisms: co-location (<1ms latency), predictive models (detect informed flow patterns), fast cancels (<100μs)',
      'Queue position: cancel and rejoin when pushed back in queue (signals informed flow building)',
      'Spread widening: increase from 5bps to 20bps during earnings, Fed announcements, high volatility',
    ],
  },
  {
    id: 'ats-7-1-q-3',
    question:
      'Design inventory risk management system: (1) Position limits, (2) Quote skewing algorithm, (3) Hedging rules. Calculate maximum loss if inventory stuck at limit during 5% price move.',
    sampleAnswer: `**Inventory Risk Management**: (1) Limits: max ±$100K position, (2) Skewing: shift quotes 0.5bps per $10K inventory (at $100K, quotes shifted 5bps), (3) Hedging: hedge 50% with futures when inventory >$50K, 100% hedge at $75K. Max loss at $100K inventory during 5% adverse move: $100K × 5% = $5,000 (before hedging); with 100% hedge: $0.`,
    keyPoints: [
      'Position limits: absolute max ±$100K, soft limit ±$50K (start hedging), alerts at ±$25K',
      'Quote skewing: linear skew 0.5bps per $10K inventory; at max inventory (+$100K), bid/ask both shift +5bps (discourages more buys)',
      'Hedging rules: dynamic hedging: 0-50K unhedged, 50-75K hedge 50%, >75K hedge 100%; use futures for fast execution',
      'Max loss calculation: $100K inventory × 5% move = $5K loss without hedge; with 100% hedge: $0 (hedged)',
      'Additional protection: hard stop at ±$125K (emergency liquidation), spread widening at ±$75K',
    ],
  },
];
