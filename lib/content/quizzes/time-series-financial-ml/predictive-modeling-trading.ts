export const predictiveModelingTradingQuiz = [
  {
    id: 'pmt-q-1',
    question:
      'Design a complete ML trading pipeline: feature engineering, model selection, validation, and deployment. How do you prevent overfitting and ensure robustness?',
    sampleAnswer:
      'Pipeline: (1) Features: 50+ technical/fundamental features, (2) Walk-forward validation, (3) Ensemble models, (4) Regular retraining. Prevent overfitting: Limit features (<50), use regularization, walk-forward not random split, monitor train vs test gap. Expected 52-58% accuracy, Sharpe 0.8-1.2.',
    keyPoints: [
      '50+ features: technical indicators + volume + volatility',
      'Walk-forward validation: train 252 days, test 21 days',
      'Ensemble: XGBoost + RF + Logistic weighted',
      'Retrain every 21 days, monitor overfitting',
      'Expected: 55% accuracy, Sharpe 1.0',
    ],
  },
  {
    id: 'pmt-q-2',
    question:
      'Compare classification (predict direction) vs regression (predict magnitude) for trading. Which is more practical and why?',
    sampleAnswer:
      'Classification easier and more reliable. Predicting exact returns (regression) very hard (R²<0.15). Predicting direction (up/down/sideways) achieves 55-60% accuracy. Use classification for signals, regression for rough magnitude estimation. Practical: Classify direction, then use fixed position sizes or Kelly criterion.',
    keyPoints: [
      'Classification: 55-60% accuracy, more reliable',
      'Regression: R²<0.15, too noisy for exact predictions',
      'Use classification for signals, fixed sizing',
      'Three-class: up/sideways/down better than binary',
      'Combine: classify direction + estimate magnitude',
    ],
  },
  {
    id: 'pmt-q-3',
    question:
      'Your model has 70% accuracy in backtesting but only 52% in live trading. Diagnose potential issues and fixes.',
    sampleAnswer:
      'Issues: (1) Overfitting: too many features, too complex model. (2) Lookahead bias: features use future data. (3) Survivorship bias: backtested on current stocks only. (4) Market regime change: trained on bull market, tested in bear. Fixes: Simplify model, verify no lookahead, test on delisted stocks, use regime-adaptive model. Re-validate with walk-forward.',
    keyPoints: [
      'Overfitting: reduce features, simplify model',
      'Lookahead bias: check all features are lagged',
      'Survivorship bias: test on historical universe',
      'Regime change: train on multiple market conditions',
      'Walk-forward validation catches most issues',
    ],
  },
];
