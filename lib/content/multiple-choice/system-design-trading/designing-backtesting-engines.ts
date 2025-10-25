import { MultipleChoiceQuestion } from '@/lib/types';

export const designingBacktestingEnginesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'dbe-mc-1',
      question:
        'What is the primary advantage of event-driven backtesting over vectorized backtesting?',
      options: [
        'Event-driven is much faster to execute',
        'Event-driven prevents lookahead bias by design',
        'Event-driven requires less memory',
        'Event-driven is easier to implement',
      ],
      correctAnswer: 1,
      explanation:
        'Event-driven prevents lookahead bias because strategy only has access to past data at each timestamp—future data physically not available. Vectorized backtesting loads entire dataset in memory, making it easy to accidentally use future data (e.g., forgetting .shift()). Event-driven is actually slower (loops vs vectorization) and more complex to implement. Memory benefit exists but secondary. Primary advantage is correctness, not performance.',
    },
    {
      id: 'dbe-mc-2',
      question:
        'In walk-forward optimization, what does it indicate if a strategy requires re-optimization every month to maintain performance?',
      options: [
        'The strategy is highly profitable and adaptive',
        'The strategy is likely overfit to market noise',
        'The strategy has low transaction costs',
        'The strategy is suitable for high-frequency trading',
      ],
      correctAnswer: 1,
      explanation:
        "Frequent re-optimization (monthly) suggests overfitting to recent noise rather than capturing genuine market patterns. Robust strategies should work with parameters optimized years ago—market fundamentals don't change monthly. Needing constant re-optimization means the strategy has no lasting edge. High transaction costs from frequent parameter changes kill profitability. Good strategies work across time periods with stable parameters.",
    },
    {
      id: 'dbe-mc-3',
      question: 'Why is survivorship bias a serious problem in backtesting?',
      options: [
        'It causes the backtest to run slower due to missing data',
        'It inflates returns by excluding bankrupt/delisted companies',
        'It requires more sophisticated data storage systems',
        'It makes parallelization of backtests difficult',
      ],
      correctAnswer: 1,
      explanation:
        "Survivorship bias inflates returns: testing only on stocks that survived (exist today) misses all failures (Lehman Brothers, Enron, etc.). Example: Testing 2000-2020 only on 2020 survivors ignores companies that went to $0. Causes backtest to overestimate returns by 2-5%+ annually. Solution: Use point-in-time universe including delisted stocks. Has nothing to do with speed, storage, or parallelization—it's a data selection bias.",
    },
    {
      id: 'dbe-mc-4',
      question:
        'For a strategy with 50 trades per year at 0.15% cost per trade, what is the annual cost drag?',
      options: ['1.5%', '3.75%', '7.5%', '15%'],
      correctAnswer: 2,
      explanation:
        'Annual cost = trades_per_year × cost_per_trade = 50 × 0.15% = 7.5%. This is substantial: a strategy with 15% gross return becomes 7.5% net return (50% reduction). Many backtest strategies fail in live trading because they underestimate transaction costs. Rule of thumb: 0.15% per trade (buy or sell), so round trip (buy + sell same stock) = 0.30%. 50 trades = 25 round trips if trading same stocks.',
    },
    {
      id: 'dbe-mc-5',
      question:
        'What is the correct way to split time series data for strategy development?',
      options: [
        'Random 80/20 split like traditional ML',
        'Split by symbol: some symbols for train, others for test',
        'Time-based: train on past data, test on future data',
        'K-fold cross-validation with random folds',
      ],
      correctAnswer: 2,
      explanation:
        "Time-based split is critical: train on past (2000-2019), test on future (2020-2024). Respects temporal causality—can't use future to predict past. Random split violates causality (training on 2023 data, testing on 2020). Splitting by symbol doesn't test time robustness. K-fold with random folds also violates causality. Time series requires TimeSeriesSplit: train on expanding window, test on next period. Only time-based split realistic for trading.",
    },
  ];
