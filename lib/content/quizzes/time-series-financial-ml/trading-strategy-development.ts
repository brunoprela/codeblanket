export const tradingStrategyDevelopmentQuiz = [
  {
    id: 'tsd-q-1',
    question:
      'Compare momentum vs mean reversion strategies. When does each work best?',
    sampleAnswer:
      'Momentum: Buy winners, ride trends. Works in trending markets (bull/bear), news-driven moves, longer timeframes (weeks-months). Mean Reversion: Fade extremes, return to average. Works in range-bound markets, high volatility, shorter timeframes (hours-days). Markets alternate: trending 30%, mean-reverting 40%, random 30%. Combine both or detect regime.',
    keyPoints: [
      'Momentum: trend-following, buy winners',
      'Mean reversion: fade extremes, buy oversold',
      'Momentum works in trends, MR in ranges',
      'Combine: use momentum for direction, MR for entry',
      'Regime detection: HMM to switch strategies',
    ],
  },
  {
    id: 'tsd-q-2',
    question:
      'Design a complete ML trading strategy: data, features, model, signals, execution.',
    sampleAnswer:
      'Pipeline: (1) Data: OHLCV + technical indicators, (2) Features: 50+ (RSI, MACD, volatility, volume), (3) Model: XGBoost predict direction, (4) Signals: Predict==1 → Long, ==-1 → Short, (5) Filters: Only trade if volatility < threshold, volume > min, (6) Position sizing: Kelly criterion or fixed %, (7) Stops: 2% stop-loss, 5% take-profit, (8) Walk-forward validation. Expected: 55% accuracy, Sharpe 1.0.',
    keyPoints: [
      'Features: technical + fundamental + sentiment',
      'Model: XGBoost or ensemble (55-60% accuracy)',
      'Filters: volatility, volume, liquidity',
      'Position sizing: Kelly or fixed % (2-5%)',
      'Risk management: stop-loss, take-profit, max drawdown',
    ],
  },
  {
    id: 'tsd-q-3',
    question:
      'How do you combine multiple signals (momentum + mean reversion + ML)?',
    sampleAnswer:
      'Ensemble signals: (1) Independent: Each signal generates trades, combine returns. (2) Weighted: 50% momentum + 30% ML + 20% MR. (3) Conditional: Use momentum in trends (ADX>25), MR in ranges (ADX<20). (4) Machine learning: Train meta-model on signal predictions. Best: Regime-based switching. Detect trend/range, use appropriate strategy. Improves Sharpe 20-30% vs single strategy.',
    keyPoints: [
      'Weighted average: 50% momentum + 30% ML + 20% MR',
      'Regime-based: switch strategy by market condition',
      'Meta-model: train on individual signal predictions',
      'Improves Sharpe 20-30% vs single signal',
      'Correlation: Combine uncorrelated signals (momentum + MR)',
    ],
  },
];
