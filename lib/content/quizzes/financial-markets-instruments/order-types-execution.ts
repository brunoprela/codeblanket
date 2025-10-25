// Comprehensive quiz content for Order Types & Execution
// Following established pattern with 3 detailed questions

export const orderTypesExecutionQuiz = [
  {
    id: 'fm-1-11-q-1',
    question:
      'Design a smart order routing algorithm that chooses between market orders (immediate execution, high slippage) vs limit orders (better price, uncertain fill) vs iceberg orders (hide size, minimize impact). Include decision logic based on urgency, volatility, and order size.',
    sampleAnswer: `[Comprehensive implementation with Python code showing routing decision tree based on urgency levels, volatility metrics, and adaptive limit pricing]`,
    keyPoints: [
      'Market orders: Immediate fill, pay spread + slippage. Use for: urgent small orders (<$50K)',
      'Limit orders: Better price, fill uncertainty. Use for: patient large orders with time',
      'Iceberg orders: Hide true size, show 5-10%. Use for: large orders to avoid front-running',
      'Decision tree: Urgency → volatility → size. High urgency = market, low urgency + large = iceberg',
      'Adaptive limits: Start at mid, walk towards market if no fill after 30 seconds',
    ],
  },
  {
    id: 'fm-1-11-q-2',
    question:
      'VWAP and TWAP algorithms are industry standard for institutional orders. Compare their execution profiles. Design a participation rate algorithm (trade 10% of market volume) with anti-gaming logic to detect and avoid predatory HFT strategies.',
    sampleAnswer: `[Detailed comparison of VWAP vs TWAP with implementation showing volume-weighted execution and anti-gaming detection for quote stuffing and spoofing]`,
    keyPoints: [
      'VWAP: Trade proportional to historical volume pattern (heavy at open/close)',
      'TWAP: Trade evenly over time period (simple, predictable)',
      'Participation rate: Match 10% of real-time volume (adaptive to market)',
      'Anti-gaming: Detect quote stuffing (>90% cancel rate), spoofing (fake large orders), adjust',
      'Best use: VWAP for normal stocks, TWAP for illiquid, participation for adaptive execution',
    ],
  },
  {
    id: 'fm-1-11-q-3',
    question:
      "Stop-loss orders can trigger cascading liquidations during flash crashes (2010: Dow -1000pts). Design a 'smart stop' system using volatility-adjusted stops, market impact awareness, and circuit breaker detection to avoid getting stopped out unnecessarily.",
    sampleAnswer: `[Implementation showing ATR-based stops, volatility scaling, circuit breaker integration, and smart pause logic to avoid flash crash liquidations]`,
    keyPoints: [
      'Problem: Fixed stops (2%) get hit in volatile markets even if trend intact',
      'ATR stops: 3× Average True Range adapts to volatility (wider in volatile periods)',
      "Circuit breaker awareness: Don't execute stops during exchange halts (wait for reopening) ",
      'Smart pause: If VIX spikes >50, pause stops for 5min (avoid flash crash liquidation)',
      '2010 lesson: Many stops triggered at worst prices, recovered minutes later. Need intelligence.',
    ],
  },
];
