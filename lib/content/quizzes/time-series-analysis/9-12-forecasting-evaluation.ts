export const forecastingEvaluationQuiz = [
  {
    id: 1,
    question:
      "Your team built a stock return forecasting model with 65% direction accuracy (vs 50% random). The portfolio manager is skeptical: 'This is only slightly better than a coin flip. Is 65% enough to be profitable after transaction costs?' Analyze: (1) Statistical significance of 65% (hypothesis test), (2) Required accuracy for profitability given 5bps transaction costs, (3) Sharpe ratio implications, (4) Position sizing strategies to maximize edge, and (5) How accuracy requirements change with leverage.",
    answer: `[Analysis: 65% with n=1000 forecasts → z-test significant (p < 0.001); Break-even accuracy with 5bps costs ≈ 51-52% → 65% provides ~13% edge; Sharpe ≈ (0.65-0.50) × sqrt(252) ≈ 2.4 if fully invested; Kelly criterion: f* ≈ 2×(0.65-0.50) = 30% per trade; With 10x leverage: need > 60% accuracy to cover higher effective costs; Real-world: 65% direction accuracy is excellent, most quant funds achieve 52-58%.]`,
  },
  {
    id: 2,
    question:
      "Design a forecast combination strategy that averages predictions from ARIMA, GARCH, VAR, and ML models. Should weights be equal, or optimized? If optimized, what objective function? Address: out-of-sample weight selection, regime-dependent weights, robustness to model mis-specification, and implementation in production system.",
    answer: `[Strategy: (1) Equal weights (1/N rule) surprisingly robust - Timmermann's finding; (2) Optimal weights via minimum forecast error variance - requires covariance matrix; (3) Time-varying weights using exponential smoothing on past performance; (4) Regime-dependent: GARCH higher weight in volatile periods, ML in stable; Out-of-sample: estimate weights on validation set, not test; Production: rolling 3-month weight updates, constrain to [0, 0.5] per model for diversification; Empirical: ensemble often beats individual models due to diversification.]`,
  },
  {
    id: 3,
    question:
      "Critique this forecast evaluation: 'My model has 0.82 correlation with actual returns and R² = 0.67 in backtest, so it captures 67% of return variation!' Explain why correlation and R² can be misleading for time series forecasting. Propose better metrics for: (1) Forecast accuracy, (2) Trading profitability, (3) Risk management, (4) Economic value.",
    answer: `[Critique: Correlation measures association not prediction; R² in-sample overstates out-of-sample; Both ignore timing (lag) and direction; Better metrics: (1) Accuracy: Out-of-sample RMSE, MAE, direction accuracy, (2) Trading: Sharpe ratio of strategy, max drawdown, hit ratio, (3) Risk: Coverage of prediction intervals, VaR accuracy, (4) Economic: Utility-based, transaction-cost adjusted returns, certainty equivalent; Key: Always use out-of-sample, economically relevant metrics, not in-sample statistical fit!]`,
  },
];

