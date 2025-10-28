export const performanceMetricsForTradingQuiz = [
  {
    id: 1,
    question:
      'What is the primary difference between Sharpe Ratio and Sortino Ratio?',
    options: [
      'Sharpe uses standard deviation, Sortino uses downside deviation',
      'Sharpe is annualized, Sortino is not',
      'Sortino is always higher than Sharpe',
      'There is no difference',
      'Sortino includes transaction costs',
    ],
    correctAnswer: 0,
    explanation:
      'Sortino Ratio only penalizes downside volatility (negative returns), while Sharpe Ratio penalizes all volatility. This makes Sortino more appropriate for strategies with asymmetric return distributions.',
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'A strategy has 60% win rate with average win of $500 and average loss of $1000. What is the expectancy?',
    options: [
      '$100 loss per trade',
      '$0 (breakeven)',
      '$100 profit per trade',
      '$200 profit per trade',
      'Cannot determine without more information',
    ],
    correctAnswer: 2,
    explanation:
      'Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss) = (0.6 × 500) - (0.4 × 1000) = 300 - 400 = -$100. This strategy loses money on average despite 60% win rate!',
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question: 'What does a Calmar Ratio of 2.0 indicate?',
    options: [
      '2% annual return',
      '2x leverage',
      'Annual return is 2x the maximum drawdown',
      'Strategy has 2:1 profit factor',
      'Sharpe ratio is 2.0',
    ],
    correctAnswer: 2,
    explanation:
      'Calmar Ratio = Annual Return / Max Drawdown. A ratio of 2.0 means annual return is twice the worst drawdown. Higher is better - indicates good return per unit of drawdown risk.',
    difficulty: 'easy',
  },
  {
    id: 4,
    question:
      'Your strategy has Sharpe 1.5 in backtest, 1.0 in paper trading, 0.5 in live trading. What does this degradation suggest?',
    options: [
      'The strategy was overfit',
      'Transaction costs are higher than expected',
      'Execution slippage is significant',
      'All of the above are likely contributors',
      'This is normal and expected',
    ],
    correctAnswer: 3,
    explanation:
      'Progressive degradation from backtest → paper → live indicates multiple issues: overfitting (backtest too optimistic), transaction costs (paper trading reveals), execution problems (live trading reveals). This is why multi-stage validation is critical.',
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'What is the minimum Sharpe Ratio generally considered acceptable for a production trading strategy?',
    options: [
      '0.5',
      '1.0',
      '1.5',
      '2.0',
      'It depends on the strategy type and risk tolerance',
    ],
    correctAnswer: 4,
    explanation:
      'There is no universal threshold - it depends on context. High-frequency: target 2.0+, Daily strategies: 1.0+ acceptable, Long-term: 0.5+ may work. Also consider: drawdown tolerance, capital constraints, opportunity cost. Focus on risk-adjusted returns relative to alternatives.',
    difficulty: 'advanced',
  },
];

export const performanceMetricsForTradingDiscussion = [
  {
    id: 1,
    question:
      'Explain why a strategy with high win rate (80%) can still lose money, and design a metric that would catch this issue early.',
    answer:
      "High win rate doesn't guarantee profitability if losses are larger than wins. Example: 80% win rate, $100 average win, $500 average loss = -$20 expectancy. Key metric: Profit Factor = Gross Profits / Gross Losses. Must be > 1.0. Or use Expectancy = (WinRate × AvgWin) - (LossRate × AvgLoss). Must be positive.",
  },
  {
    id: 2,
    question:
      'Design a comprehensive performance dashboard that compares multiple strategies across all key metrics. What visualizations would you include and why?',
    answer:
      'Include: 1) Scatter plot of Sharpe vs Max Drawdown, 2) Equity curves comparison, 3) Monthly returns heatmap, 4) Drawdown periods visualization, 5) Metrics table (Sharpe, Sortino, Calmar, Win Rate, Profit Factor), 6) Rolling Sharpe to detect degradation, 7) Return distribution histograms. This provides complete performance picture.',
  },
  {
    id: 3,
    question:
      'Your strategy shows Sharpe 2.5 in backtest but only 1.2 in walk-forward analysis. Explain the likely causes and what this means for deployment.',
    answer:
      'Causes: 1) Overfitting - parameters optimized for specific period, 2) Regime change - market conditions different in walk-forward period, 3) Data mining - tested many strategies, 4) Look-ahead bias in backtest. For deployment: Use walk-forward Sharpe (1.2) as realistic expectation. Expect further degradation to 0.8-1.0 in live trading. May still be viable if risk-adjusted.',
  },
];
