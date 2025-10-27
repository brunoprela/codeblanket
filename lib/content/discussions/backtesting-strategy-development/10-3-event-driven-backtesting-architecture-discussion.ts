export const eventDrivenBacktestingArchitectureDiscussion = [
  {
    id: 1,
    question:
      "Design an event-driven backtesting system that can handle multiple timeframes simultaneously (1-minute bars for execution, daily bars for signals). Explain the architecture, event handling, and how you'd prevent timing issues.",
    answer: `Multi-timeframe event-driven systems require careful timestamp management and event ordering. Use a unified event queue with precise timestamps, create separate DataHandlers for each timeframe, synchronize events using a TimeManager component, and ensure lower timeframe data is processed before higher timeframe signals at the same timestamp.`,
  },
  {
    id: 2,
    question:
      'Your event-driven backtest engine needs to support both historical backtesting and live trading with minimal code changes. How would you design the system to enable this? What interfaces and abstractions would you create?',
    answer: `Create abstract interfaces for DataHandler (historical vs live feed), ExecutionHandler (simulated vs broker API), and Portfolio (tracking only). Implement concrete classes for each mode. Use dependency injection to swap implementations. The Strategy and core event loop remain identical across modes.`,
  },
  {
    id: 3,
    question:
      'Implement a sophisticated ExecutionHandler that simulates realistic order fills including: partial fills based on volume, market impact, and time-in-force (IOC, GTC). How would you model each of these aspects?',
    answer: `Model partial fills by comparing order size to bar volume (fill max 10% of volume per bar). Market impact: temporary price impact = size / volume * volatility. Permanent impact: affects subsequent bars. IOC: fill immediately or cancel. GTC: keep in order book across multiple bars until filled or cancelled. Track unfilled orders in a separate queue.`,
  },
];
