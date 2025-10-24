export const orderExecutionTradingInfrastructureQuiz = [
  {
    id: 'oeti-q-1',
    question:
      'Design robust order execution system with error handling, monitoring, and failsafes.',
    sampleAnswer:
      'System: (1) Order queue: Thread-safe queue for async execution. (2) Execution engine: Submits to broker API with retry logic (3 attempts). (3) Monitoring: Track fill rate, slippage, latency. Log all orders. (4) Failsafes: Daily loss limit (-5% → stop trading), position limits (max 10), kill switch (emergency stop). (5) Reconciliation: Compare broker positions vs internal state every minute. (6) Alerts: Email/SMS on errors, large slippage. (7) Testing: Paper trading 1 month before live. Critical: Never risk more than daily limit.',
    keyPoints: [
      'Order queue + async execution',
      'Retry logic (3 attempts), error handling',
      'Monitoring: fill rate, slippage, latency',
      'Failsafes: daily loss limit, kill switch',
      'Reconciliation, alerts, paper testing first',
    ],
  },
  {
    id: 'oeti-q-2',
    question: 'What is slippage? How do you measure and minimize it?',
    sampleAnswer:
      'Slippage: Difference between expected price and actual fill. Example: Expected $100, filled $100.10 → slippage $0.10 (0.1%). Causes: (1) Market orders in illiquid stocks, (2) Large size, (3) Volatile markets. Measure: Track average slippage per trade. Minimize: (1) Use limit orders, (2) Split large orders (TWAP/VWAP), (3) Trade liquid stocks, (4) Avoid market open/close, (5) Passive orders (maker not taker). Typical: 0.05-0.1%. HFT: <0.01%. High slippage kills profitability.',
    keyPoints: [
      'Slippage = actual fill - expected price',
      'Causes: illiquid stocks, large size, volatility',
      'Minimize: limit orders, split large orders, liquid stocks',
      'Typical: 0.05-0.1%, HFT <0.01%',
      'Track and optimize - major cost for frequent traders',
    ],
  },
  {
    id: 'oeti-q-3',
    question:
      'Compare paper trading vs live trading. What changes when going live?',
    sampleAnswer:
      'Paper: Simulated, no real money, perfect fills, no slippage, no psychology. Live: Real money, slippage (0.05-0.1%), partial fills, broker errors, emotional stress. Changes: (1) Fills worse (slippage, rejection), (2) Psychology: fear/greed, panic exits, (3) Costs: commissions, market data, (4) Capacity: Large orders move market. Process: Paper trade 1-3 months → small live size (10%) → gradually scale. Most strategies perform 20-30% worse live vs paper. Always start small. Emotions biggest difference.',
    keyPoints: [
      'Paper: perfect fills, no slippage, no emotions',
      'Live: slippage, costs, partial fills, psychology',
      'Performance 20-30% worse live vs paper',
      'Process: paper 1-3 months → 10% live → scale',
      'Start small, emotions are biggest challenge',
    ],
  },
];
