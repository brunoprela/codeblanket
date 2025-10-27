export const forecastingEvaluationMultipleChoice = [
  {
    id: 1,
    question:
      'In walk-forward validation, you use expanding window (all past data) vs rolling window (fixed size). Expanding window is preferred when:',
    options: [
      'Data has structural breaks',
      'Relationship is stable and more data helps',
      'Computational speed is priority',
      'Recent data more relevant',
      'Always prefer rolling',
    ],
    correctAnswer: 1,
    explanation:
      'Relationship stable → more data helps. Expanding: Uses all historical data, estimates improve with sample size IF relationship constant. Rolling: Only recent N observations, adapts to regime changes but less data. Trade-off: Stability vs adaptability. Use expanding when: long-run stable relationship, no structural breaks, want precise estimates. Use rolling when: time-varying parameters, regime changes, recent data more predictive. Financial data: Often rolling due to regime changes, but test both!',
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'Model A has RMSE=2.0%, Model B has RMSE=2.1%. Are they significantly different?',
    options: [
      'Yes - Model A is better',
      'No - need Diebold-Mariano test',
      'Yes - 5% difference is significant',
      'Cannot determine without R²',
      'Models are equivalent',
    ],
    correctAnswer: 1,
    explanation:
      "Need Diebold-Mariano test. Point estimates (RMSE) don't show statistical significance. DM test: Compares forecast errors accounting for correlation and sample size. Null: Equal predictive accuracy. Test statistic: t = mean(d) / std(d) where d = error1² - error2². Small RMSE difference might not be significant with: small sample, highly correlated forecasts, volatile errors. Conclusion: Always test significance, don't just compare point estimates!",
    difficulty: 'advanced',
  },
  {
    id: 3,
    question:
      'Direction accuracy 60% seems good, but returns still negative after trading costs. Why?',
    options: [
      'Direction accuracy is mis-calculated',
      'Incorrect directions are large losses, correct are small gains',
      'Sample size too small',
      'Transaction costs too high',
      'Model is overfitted',
    ],
    correctAnswer: 1,
    explanation:
      'Asymmetric gains/losses. Direction accuracy alone insufficient - must consider magnitude! Example: Correct 6/10 times with +0.1% each = +0.6%, Wrong 4/10 times with -0.3% each = -1.2%, Net = -0.6% (loss!). This happens when: model catches small moves but misses large reversals, risk management poor (no stop-losses), skewness in error distribution. Better metric: Average return per trade conditional on correct vs wrong direction. Solution: Combine direction with confidence-weighted position sizing.',
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'Why is MAPE (Mean Absolute Percentage Error) problematic for return forecasting?',
    options: [
      'MAPE assumes normal distribution',
      'Division by actual returns causes issues near zero',
      'MAPE only works for prices not returns',
      'MAPE is not scale-free',
      'Computational complexity',
    ],
    correctAnswer: 1,
    explanation:
      'Division by zero problem. MAPE = mean(|forecast - actual| / |actual|) × 100. For returns: actuals often near zero → division explodes! Example: Actual = 0.001%, Forecast = 0.002% → MAPE = 100% (terrible!), but absolute error only 1bp. Alternatives: (1) MAE for returns (not percentage), (2) RMSE, (3) sMAPE (symmetric MAPE), (4) Direction accuracy. Use MAPE for: prices, volatility (always positive, away from zero). Avoid for: returns, spreads (can be near zero).',
    difficulty: 'intermediate',
  },
  {
    id: 5,
    question:
      'Model shows 95% prediction interval coverage of only 88% out-of-sample. This indicates:',
    options: [
      'Model is well-calibrated',
      'Prediction intervals too narrow (overconfident)',
      'Prediction intervals too wide',
      'Sample size too small for assessment',
      'Model is conservative',
    ],
    correctAnswer: 1,
    explanation:
      'Intervals too narrow - overconfident. Coverage = % of actuals falling within prediction intervals. Target: 95% intervals should contain 95% of actuals. Here: Only 88% contained → intervals underestimate uncertainty! Causes: (1) Underestimated error variance, (2) Model mis-specification (missing dynamics), (3) Parameter uncertainty ignored, (4) Non-normal errors with fat tails. Solutions: (1) Use bootstrap for intervals, (2) Increase interval width (multiply by factor), (3) Better model specification, (4) Heavy-tailed distributions (Student-t). Proper calibration essential for risk management!',
    difficulty: 'advanced',
  },
];
