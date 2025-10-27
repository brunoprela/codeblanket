export const eventDrivenBacktestingArchitectureQuiz = [
  {
    id: 1,
    question:
      'What is the main advantage of event-driven backtesting over vectorized backtesting?',
    options: [
      'Event-driven backtesting is much faster and more efficient',
      'Event-driven backtesting can more realistically simulate order execution, partial fills, and complex portfolio logic',
      'Event-driven backtesting is easier to implement',
      'Event-driven backtesting uses less memory',
      'Vectorized backtesting cannot handle multiple stocks',
    ],
    correctAnswer: 1,
    explanation:
      'Event-driven backtesting processes data point-by-point, simulating exactly how live trading works. This allows realistic modeling of: order execution delays, partial fills, market impact, complex portfolio logic, and dynamic risk management. Vectorized backtesting is faster but cannot easily model these realistic execution details. Production systems use event-driven because the backtest code can be reused for live trading with minimal changes.',
    difficulty: 'beginner',
  },
  {
    id: 2,
    question:
      'In an event-driven backtest, what is the correct order of event processing?',
    options: [
      'Market → Fill → Signal → Order',
      'Signal → Market → Order → Fill',
      'Market → Signal → Order → Fill',
      'Order → Signal → Market → Fill',
      'The order does not matter',
    ],
    correctAnswer: 2,
    explanation:
      'Correct flow: 1) **Market Event**: New bar data arrives, 2) **Signal Event**: Strategy evaluates data and generates signal, 3) **Order Event**: Portfolio converts signal to order with position sizing, 4) **Fill Event**: Execution handler simulates order execution and creates fill. This mirrors real trading where you receive market data, make trading decision, place order, and receive fill confirmation.',
    difficulty: 'beginner',
  },
  {
    id: 3,
    question:
      'You are building an event-driven backtest engine. Where should you implement position sizing logic?',
    options: [
      'In the Strategy class when generating signals',
      'In the Portfolio class when converting signals to orders',
      'In the ExecutionHandler when filling orders',
      'In the DataHandler when streaming bars',
      'Position sizing should be handled separately from the backtest',
    ],
    correctAnswer: 1,
    explanation:
      'Position sizing belongs in the **Portfolio class**. The Strategy should only generate signals (BUY/SELL/EXIT with strength). The Portfolio then: checks available cash, calculates position size based on risk management rules, creates appropriately sized orders. This separation of concerns makes strategies reusable across different portfolio configurations. Example: Same strategy can be used with 100% capital per trade or 10% capital per trade by changing Portfolio settings only.',
    difficulty: 'intermediate',
  },
  {
    id: 4,
    question:
      'What is the purpose of using a PriorityQueue for events in an event-driven backtest?',
    options: [
      'To process events faster',
      'To ensure events are processed in chronological order, preventing look-ahead bias',
      'To reduce memory usage',
      'To allow parallel processing of events',
      'PriorityQueue is not necessary - regular Queue works fine',
    ],
    correctAnswer: 1,
    explanation:
      "PriorityQueue ensures events are processed in **chronological order** based on timestamp. This is critical because: 1) Prevents look-ahead bias (can't process tomorrow's data before today's), 2) Maintains realistic timeline of events, 3) Handles situations where events arrive out of order. Without proper ordering, you might process an order before the market data that triggered it, creating unrealistic backtest results. The priority is typically the event timestamp.",
    difficulty: 'intermediate',
  },
  {
    id: 5,
    question:
      'Your event-driven backtest shows 40% annual returns, but when you deploy the same code to paper trading, it generates only 15% returns. What is the MOST likely issue?',
    options: [
      'The backtest had look-ahead bias in the data handler',
      'The execution handler in backtest was too optimistic (did not model slippage, partial fills, etc.)',
      'The strategy was overfit to the training data',
      'Market conditions changed between backtest and paper trading',
      'Transaction costs were not included in backtest',
    ],
    correctAnswer: 1,
    explanation:
      'The most likely culprit is **execution simulation being too optimistic**. Common issues:\n\n1. Backtest assumes instant fills at close price\n2. Reality: Orders experience slippage, especially on large orders\n3. Backtest assumes full fills\n4. Reality: Large orders may be partially filled or split across multiple bars\n5. Backtest ignores market impact\n6. Reality: Your orders move the market\n\nExample: Backtest shows buying 10,000 shares at $100. Reality: You get 3,000 at $100, 4,000 at $100.05, 3,000 at $100.10. Average fill: $100.05 vs expected $100.\n\nFix: Implement realistic ExecutionHandler with slippage, partial fills, and market impact models. Professional quant funds spend significant effort on execution simulation.',
    difficulty: 'advanced',
  },
];
