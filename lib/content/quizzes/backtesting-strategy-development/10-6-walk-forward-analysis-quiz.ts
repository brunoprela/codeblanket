import { MultipleChoiceQuestion } from '@/lib/types';

const walkForwardAnalysisQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'wfa-1',
    question:
      'A quantitative analyst runs a backtest with parameter optimization on 5 years of data (2018-2022) and achieves a Sharpe ratio of 2.1. When testing the same strategy with those parameters on 2023 data (not used in optimization), the Sharpe ratio is 0.8. What is the most likely explanation?',
    options: [
      'Market conditions changed dramatically in 2023',
      'The strategy was overfitted to the 2018-2022 data',
      'Transaction costs were underestimated in the backtest',
      'The data quality was poor in 2023',
    ],
    correctAnswer: 1,
    explanation:
      "The significant degradation from Sharpe 2.1 to 0.8 when moving from in-sample to out-of-sample data is a classic sign of overfitting. The parameters were optimized to work perfectly on the historical data but don't generalize well to unseen data. While market conditions do change, a drop this severe typically indicates the strategy learned noise rather than true signal. Walk-forward analysis would have detected this by showing consistently poor out-of-sample performance across multiple windows.",
    difficulty: 'intermediate',
  },
  {
    id: 'wfa-2',
    question:
      'In walk-forward analysis, you have 10 years of data. You use 2 years for training and 3 months for testing, stepping forward by 3 months each time. Approximately how many walk-forward windows will you have?',
    options: ['10 windows', '20 windows', '30 windows', '40 windows'],
    correctAnswer: 2,
    explanation:
      'With 10 years of data, starting from year 2 (need 2 years for first training window), you have 8 years remaining for testing. With 3-month test periods and 3-month steps, you get approximately 32 windows (8 years × 4 quarters per year). The closest answer is 30 windows. The calculation: First window trains on years 0-2, tests on year 2.0-2.25. Second window trains on years 0.25-2.25, tests on year 2.25-2.5, and so on. This gives (10 years - 2 year training) / 0.25 year step = 32 windows.',
    difficulty: 'intermediate',
  },
  {
    id: 'wfa-3',
    question:
      'What is the primary advantage of using anchored (expanding) windows instead of rolling (fixed-size) windows in walk-forward analysis?',
    options: [
      'Anchored windows execute faster due to less data processing',
      'Anchored windows provide more stable parameter estimates by using all available historical data',
      'Anchored windows adapt more quickly to regime changes in the market',
      'Anchored windows are less prone to overfitting',
    ],
    correctAnswer: 1,
    explanation:
      "Anchored (expanding) windows use all data from the start to the current point, continuously growing the training set. This provides more stable parameter estimates because they're based on more data, reducing estimation error. However, this stability comes at a cost: anchored windows are SLOWER to adapt to regime changes because they give equal weight to old data. Rolling windows adapt faster but have higher parameter variance. Anchored windows are NOT less prone to overfitting (they can memorize more patterns from the larger dataset) and are NOT faster to execute (they process more data).",
    difficulty: 'advanced',
  },
  {
    id: 'wfa-4',
    question:
      'A walk-forward analysis shows that optimal parameters change dramatically from window to window (coefficient of variation > 100%). What does this indicate about the strategy?',
    options: [
      'The strategy is robust and adapts well to changing markets',
      'The strategy likely has unstable performance and the optimal parameters are not reliable',
      'The training period is too long and should be shortened',
      'This is normal and expected behavior for all trading strategies',
    ],
    correctAnswer: 1,
    explanation:
      "High parameter instability (CV > 100%) indicates the optimization is finding very different 'optimal' values in each window, suggesting the performance surface is flat or noisy. This means: (1) the parameters don't have a strong, consistent relationship with performance, (2) the strategy may be curve-fitting to noise, or (3) the strategy is not robust to parameter choices. A robust strategy should have relatively stable optimal parameters across windows. This is NOT a sign of adaptability—true adaptation shows parameters smoothly tracking regime changes, not jumping erratically. Shortening the training period would likely make this worse, not better.",
    difficulty: 'advanced',
  },
  {
    id: 'wfa-5',
    question:
      'In production, how frequently should you typically reoptimize strategy parameters using walk-forward analysis for a daily mean-reversion equity strategy?',
    options: [
      'Daily, to capture the most recent market dynamics',
      'Weekly, to balance adaptation with stability',
      'Monthly to quarterly, allowing enough new data to accumulate for meaningful reoptimization',
      'Annually, to avoid overfitting to short-term noise',
    ],
    correctAnswer: 2,
    explanation:
      'For a daily mean-reversion strategy, monthly to quarterly reoptimization is typically optimal. This frequency provides enough new data points (20-60 trading days) for statistically meaningful optimization while avoiding the pitfalls of overfitting to very recent noise. Daily reoptimization would be overfitting to noise and computationally expensive. Weekly is too frequent for daily strategies (only 5 new data points). Annual reoptimization is too infrequent—markets can change significantly in a year. The exact frequency depends on strategy characteristics: higher-frequency strategies may need more frequent reoptimization, while lower-frequency strategies can go longer between optimizations. Most professional shops reoptimize quarterly.',
    difficulty: 'intermediate',
  },
];

export default walkForwardAnalysisQuiz;
