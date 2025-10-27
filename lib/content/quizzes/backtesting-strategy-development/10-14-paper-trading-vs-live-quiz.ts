import { MultipleChoiceQuestion } from '@/lib/types';

const paperTradingVsLiveQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'paper-1',
    question:
      'Your strategy shows a 2.5 Sharpe ratio in backtesting. After 3 months of paper trading, it shows a 1.2 Sharpe. What is the most likely cause?',
    options: [
      'Bad luck - paper trading period was unfavorable',
      'Backtesting had unrealistic assumptions about execution costs and slippage',
      'Paper trading data feed is defective',
      'The strategy needs more optimization',
    ],
    correctAnswer: 1,
    explanation:
      'A ~52% performance degradation from backtest to paper trading typically indicates unrealistic backtest assumptions, most commonly around execution quality. Backtests often assume perfect fills at mid-price with minimal slippage. Real markets have bid-ask spreads, slippage, and market impact. Option A (bad luck) is possible but less likely for such a large drop. Option C would show data errors, not consistent underperformance. Option D is dangerous—optimizing after seeing forward results causes overfitting. This degradation is normal and why paper trading is essential. Renaissance Technologies expects 30-50% degradation from backtest to live.',
    difficulty: 'intermediate',
  },
  {
    id: 'paper-2',
    question:
      'What is the minimum recommended duration for paper trading before deploying real capital for a daily trading strategy?',
    options: [
      '1 week (sufficient to see strategy in action)',
      '1 month (covers most market conditions)',
      '3-6 months minimum (captures different market regimes)',
      'Paper trading is optional if backtesting is thorough',
    ],
    correctAnswer: 2,
    explanation:
      "3-6 months minimum for daily strategies ensures exposure to various market conditions (trending, ranging, volatile, calm). One week (Option A) is far too short. One month (Option B) might miss important regimes. Option D is dangerous—even Renaissance Technologies, with the world's most sophisticated backtesting, paper trades for 6-12 months. The goal is to observe the strategy across: earnings seasons, Fed announcements, market corrections, low-volatility periods, and high-volatility periods. For higher-frequency strategies (intraday), 1-2 months may suffice. For lower-frequency (weekly/monthly), 12+ months recommended.",
    difficulty: 'beginner',
  },
  {
    id: 'paper-3',
    question:
      "In paper trading, your strategy's market orders consistently execute at worse prices than the last traded price. Your slippage averages 8 basis points. What does this indicate?",
    options: [
      'The paper trading system is broken and overestimating slippage',
      'This is realistic - market orders execute at ask (buying) or bid (selling), which are worse than last price',
      'You should use limit orders exclusively to eliminate slippage',
      '8 bps slippage means the strategy cannot be profitable',
    ],
    correctAnswer: 1,
    explanation:
      'This is normal and realistic! Market orders execute at the ask price (when buying) or bid price (when selling), not the last traded price. The bid-ask spread for liquid stocks is typically 2-10 basis points, plus additional slippage from market impact. Option A is wrong—8 bps is realistic for retail execution. Option C is problematic because limit orders risk non-execution, potentially missing profitable trades. Option D is false—8 bps is manageable if the strategy generates sufficient alpha (>50 bps per trade). This is exactly why paper trading exists—to measure real-world execution costs that backtests often underestimate.',
    difficulty: 'advanced',
  },
  {
    id: 'paper-4',
    question: 'Your paper trading system should use which type of market data?',
    options: [
      'Historical data (same as backtesting for consistency)',
      'Delayed data (15-minute delay is sufficient)',
      'Real-time data (same data feed as live trading will use)',
      "End-of-day data only (daily strategies don't need intraday)",
    ],
    correctAnswer: 2,
    explanation:
      "Paper trading MUST use real-time market data identical to what live trading will use. Option A defeats the purpose—using historical data is just another backtest. Option B (delayed data) introduces different execution prices than live trading. Option D misses intraday volatility and execution dynamics even for daily strategies. The entire point of paper trading is forward validation with real market conditions, including: real-time price discovery, actual spreads, genuine market impact, live data feed latency, and real market microstructure. Use the exact same data infrastructure you'll use for live trading.",
    difficulty: 'beginner',
  },
  {
    id: 'paper-5',
    question:
      "During paper trading, you discover your strategy's live Sharpe (1.8) exceeds the backtest Sharpe (1.5). What should you do?",
    options: [
      'Celebrate and immediately deploy with full capital',
      'Investigate carefully - outperformance is suspicious and may indicate bugs',
      'Increase position sizes to capitalize on better-than-expected performance',
      'The backtest was too conservative; no action needed',
    ],
    correctAnswer: 1,
    explanation:
      "Outperformance in paper trading vs backtesting is a RED FLAG requiring investigation. Typical causes: (1) Look-ahead bias in paper trading system, (2) Data snooping (strategy accidentally optimized on recent data), (3) Lucky period that won't persist, (4) Bug in paper trading execution simulation. Paper trading should perform WORSE than backtesting due to real-world frictions. Options A and C are reckless. Option D is naive. Professional protocol: verify no bugs, extend paper trading period, investigate if outperformance persists, only then cautiously deploy minimal capital. Two Sigma famously killed a strategy when paper trading beat backtesting—investigation revealed a subtle data leak.",
    difficulty: 'advanced',
  },
];

export default paperTradingVsLiveQuiz;
