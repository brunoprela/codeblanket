export const vectorAutoregressionQuiz = [
  {
    id: 1,
    question:
      "Your macro trading desk uses a VAR(4) model with 6 variables (GDP growth, inflation, unemployment, 10Y yield, S&P 500, USD index) to forecast and generate trading signals. The model shows: (1) S&P 500 Granger-causes GDP (p=0.02), (2) GDP does NOT Granger-cause S&P 500 (p=0.45), (3) Impulse response shows 1% S&P shock → +0.3% GDP growth over 12 months. The economist argues: 'This proves stock market drives the economy!' The statistician counters: 'Granger causality ≠ true causality. This could be spurious.' Adjudicate this debate by explaining: (1) What Granger causality actually tests (predictive vs true causality), (2) Alternative explanations for the result (omitted variables, common factors), (3) How to test if the relationship is genuine (out-of-sample, economic theory), and (4) Whether this finding is actionable for trading.",
    answer: `[Detailed answer covering: Granger causality only tests if past X helps predict Y, not true causation; Alternative explanations: (a) forward-looking market anticipates GDP, (b) omitted variable (confidence) drives both, (c) common monetary policy; Tests: out-of-sample forecasting, structural breaks, include additional variables; Trading implications: market leads economy → use for timing but not direction. Full implementation with code examples.]`,
  },
  {
    id: 2,
    question:
      'Compare forecasting S&P 500 returns using: (A) Univariate ARIMA(1,1,1), (B) Multivariate VAR(2) with S&P, VIX, and Treasury yields, (C) VECM (Vector Error Correction Model) if cointegration exists. Design a horse-race comparison including: forecast horizons (1-day, 1-week, 1-month), evaluation metrics, computational costs, and practical implementation challenges. Under what conditions would each approach win?',
    answer: `[Comprehensive outline: ARIMA wins for 1-day (simplicity, low noise), VAR wins for 1-week (captures cross-asset dynamics), VECM wins if true cointegration (incorporates equilibrium relationships); Metrics: RMSE for accuracy, direction accuracy for trading, Sharpe for strategies; ARIMA fastest, VECM slowest but most informative; Implementation: ARIMA robust, VAR needs stationarity, VECM requires cointegration testing. Code framework for comparison.]`,
  },
  {
    id: 3,
    question:
      'A VAR(5) model with 10 variables has 10 × (10×5 + 1) = 510 parameters. With only 500 daily observations, this is severely overparameterized. Propose solutions: (1) Shrinkage methods (ridge, lasso, elastic net), (2) Bayesian VAR with priors, (3) Factor-augmented VAR, (4) Structural VAR with economic restrictions. For each, explain the methodology, implementation, pros/cons, and when to use.',
    answer: `[Outline: Ridge/Lasso penalizes large coefficients, reduces overfitting, easy to implement but lacks economic interpretation; BVAR uses Minnesota prior (recent lags more important, own lags more important than cross), principled uncertainty quantification; FAVAR extracts common factors first, reduces dimensionality; SVAR imposes economic structure (instantaneous effects), interpretable but requires theory. Use cases: BVAR for forecasting, SVAR for policy analysis, FAVAR for large datasets. Implementation code for each.]`,
  },
];
