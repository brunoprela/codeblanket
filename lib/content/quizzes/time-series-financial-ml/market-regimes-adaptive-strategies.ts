export const marketRegimesAdaptiveStrategiesQuiz = [
  {
    id: 'mras-q-1',
    question: 'What are the main market regimes? How do you detect them?',
    sampleAnswer:
      '3 main regimes: (1) Trending: Persistent direction, ADX>25, momentum works. (2) Ranging: Mean-reverting, ADX<20, oscillates. (3) High Volatility: VIX>30, reduce risk. Detection: (1) HMM on returns, (2) Volatility (rolling std), (3) ADX for trend strength, (4) Correlation (decreases in crisis). Combine multiple indicators. Typical: 30% trending, 40% ranging, 30% random/volatile. Regime detection improves Sharpe 20-30%.',
    keyPoints: [
      '3 regimes: trending, ranging, high-vol',
      'Trending: ADX>25, persistent moves',
      'Ranging: ADX<20, mean-reverting',
      'Detection: HMM, volatility, ADX, VIX',
      'Improves Sharpe 20-30% by adapting strategy',
    ],
  },
  {
    id: 'mras-q-2',
    question:
      'Design adaptive trading system that switches strategies by market regime.',
    sampleAnswer:
      'System: (1) Detect regime: Calculate ADX, volatility, correlation. (2) Classify: ADX>25 → trending, ADX<20 + vol<20% → ranging, vol>30% → high-vol. (3) Strategy selection: Trending → momentum (21-day breakout), Ranging → mean-reversion (RSI oversold/overbought), High-vol → cash or reduce to 25% size. (4) Smooth transitions: 5-day buffer to avoid whipsaw. (5) Backtest each regime separately, then combined. Expected: 30% better Sharpe than single strategy.',
    keyPoints: [
      'Detect: ADX (trend), volatility, VIX',
      'Trending → momentum, Ranging → mean-reversion, High-vol → reduce risk',
      'Smooth transitions (5-day buffer)',
      'Backtest each regime separately',
      'Expected: 30% Sharpe improvement',
    ],
  },
  {
    id: 'mras-q-3',
    question: 'How do you use Hidden Markov Models (HMM) for regime detection?',
    sampleAnswer:
      'HMM: Assumes market in one of N hidden states (regimes), each with different return distribution. Train on historical returns → identifies clusters (e.g., bull, bear, sideways). Model automatically finds: State 0: mean +15%, vol 12% (bull), State 1: mean -10%, vol 25% (bear), State 2: mean +2%, vol 15% (sideways). Prediction: Given recent returns, what regime now? Use predicted regime to select strategy. Advantage: Unsupervised, data-driven. Limitation: Requires retraining, can lag regime changes.',
    keyPoints: [
      'HMM: N hidden states, each with return distribution',
      'Train on returns → identifies bull/bear/sideways',
      'Predicts current regime from recent returns',
      'Advantages: unsupervised, data-driven',
      'Limitations: requires retraining, lags actual changes',
    ],
  },
];
