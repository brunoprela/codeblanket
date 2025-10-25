export const classicalTimeSeriesModelsQuiz = [
  {
    id: 'ctsm-q-1',
    question:
      'You are building a trading system that forecasts stock returns 1 day ahead using ARIMA. Walk through the complete process: (1) determining stationarity and transformation, (2) selecting optimal (p,d,q) order using ACF/PACF and information criteria, (3) validating the model with walk-forward testing, (4) generating signals with confidence thresholds. Include specific code patterns and diagnostic checks. Why might ARIMA struggle in volatile markets?',
    sampleAnswer:
      "Complete ARIMA trading system: (1) Stationarity: Test prices with ADF (expect p > 0.05, non-stationary). Transform to returns: r_t = log(P_t/P_{t-1}). Verify returns stationary (ADF p < 0.05). (2) Order selection: Plot PACF of returns → AR order (lags outside confidence bands). Plot ACF → MA order (sharp cutoff). Example: PACF significant at lag 1,2 → try AR(2). ACF dies quickly → low MA. Start with ARIMA(2,0,1) on returns or ARIMA(2,1,1) on prices. Compare models: for p in range(0,6): for q in range(0,4): model = ARIMA(data, order=(p,1,q)).fit(), store AIC/BIC. Select lowest AIC. Typical result: ARIMA(1,1,1) or (2,1,1) for financial data. (3) Walk-forward validation: Split train/test temporally. for i in range(252, len (prices), 21): train = prices[i-252:i], fit ARIMA, forecast 21 days, calculate MAE/RMSE. Never use future data in training. (4) Signal generation: Forecast tomorrow's price, calculate predicted return = (forecast - current) / current. If predicted return > threshold (0.5%), signal = 1 (buy). If < -threshold, signal = -1 (sell). Threshold critical to filter noise. (5) Diagnostics: Check residuals with Ljung-Box (p > 0.05 = no autocorrelation). Check normality with Jarque-Bera. If residuals have patterns, model insufficient. (6) Why ARIMA struggles: Assumes linear relationships and constant parameters. Volatile markets have: regime changes (2020 crash vs 2021 rally), volatility clustering (ARCH effects), non-linear patterns. ARIMA forecasts near current value (mean-reverting). In trends, always lags. Solution: Combine ARIMA with GARCH for volatility, or use ensemble with ML models.",
    keyPoints: [
      'Transform prices to returns, verify stationarity with ADF test',
      'Use PACF for AR order, ACF for MA order, compare models with AIC/BIC',
      'Walk-forward validation: train on past 252 days, test on next 21, slide forward',
      'Generate signals with threshold (0.5%+) to filter noise',
      'ARIMA struggles with regime changes, volatility clustering, non-linear patterns',
    ],
  },
  {
    id: 'ctsm-q-2',
    question:
      'Explain the difference between AR, MA, and ARMA models from both mathematical and intuitive perspectives. Given ACF and PACF plots, how do you identify which model is appropriate? Provide specific patterns for AR(1), AR(2), MA(1), MA(2), and ARMA(1,1). What financial scenarios fit each model type?',
    sampleAnswer:
      "Model characteristics: (1) AR(p): Y_t = c + φ_1*Y_{t-1} + ... + φ_p*Y_{t-p} + ε_t. Intuition: Today depends on yesterday. Like momentum—positive returns follow positive returns. PACF: Significant at lags 1 to p, then cuts off. ACF: Exponential decay. Example: AR(1) with φ_1 = 0.3 shows weak persistence. AR(2) with φ_1 = 0.4, φ_2 = 0.2 shows 2-day memory. Financial fit: Mean-reverting assets (pairs trading spreads, VIX). (2) MA(q): Y_t = μ + ε_t + θ_1*ε_{t-1} + ... + θ_q*ε_{t-q}. Intuition: Today depends on recent shocks (news events). ACF: Significant at lags 1 to q, sharp cutoff. PACF: Exponential decay. Example: MA(1) with θ_1 = -0.3 means yesterday's shock negatively affects today. MA(2) means 2-day shock memory. Financial fit: News-driven markets (earnings surprises, Fed announcements). (3) ARMA(p,q): Combines both. ACF: Exponential decay (from AR). PACF: Exponential decay (from MA). More complex patterns. Financial fit: Most real financial series (mixture of momentum and shocks). Identification rules: If PACF cuts off at lag p, ACF decays → AR(p). If ACF cuts off at lag q, PACF decays → MA(q). If both decay gradually → ARMA(p,q). If ACF/PACF both significant at many lags → non-stationary, apply differencing. Practical example: Stock returns usually show low autocorrelation (weak ACF/PACF) → ARMA(1,1) or ARMA(2,1) sufficient. VIX shows strong AR(1) (φ ≈ 0.8-0.9) due to volatility persistence. Bond yields show MA components from monetary policy shocks.",
    keyPoints: [
      'AR: depends on past values, PACF cuts off, ACF decays (momentum/mean-reversion)',
      'MA: depends on past shocks, ACF cuts off, PACF decays (news-driven)',
      'ARMA: both components, both ACF/PACF decay (mixed patterns)',
      'Identification: PACF cutoff → AR order, ACF cutoff → MA order',
      'Financial: returns are weak ARMA, VIX is strong AR, yields have MA from policy',
    ],
  },
  {
    id: 'ctsm-q-3',
    question:
      'Your ARIMA model passes all statistical tests (stationary residuals, no autocorrelation, normal distribution) but fails in walk-forward testing with poor out-of-sample performance. Diagnose potential causes and propose solutions. Consider: overfitting, parameter instability, regime changes, and model limitations. How would you improve the forecasting system?',
    sampleAnswer:
      'Diagnosis and solutions: (1) Overfitting: Model fits training noise, not signal. Check: If in-sample R² > 0.5 but out-of-sample R² < 0.05 → overfitting. Cause: Too many parameters (high p,q). Solution: Use BIC instead of AIC (penalizes complexity more), limit max p,q to 3, use simpler models. (2) Parameter instability: Coefficients change over time (2020 crisis vs 2023 normal). Check: Rolling parameter estimates, plot φ_t over time. If volatile → instability. Solution: Shorter training window (126 days vs 252), adaptive models that re-weight recent data. (3) Regime changes: 2019 (bull) vs 2020 (crash) vs 2021 (recovery) have different dynamics. ARIMA assumes constant parameters. Check: Chow test for structural breaks. Solution: Regime-switching models (Markov-switching ARIMA), or separate models per regime. (4) Model limitations: ARIMA is linear. Financial markets are non-linear (leverage effects, jumps). Check: If large prediction errors cluster during extreme events → non-linearity. Solution: (a) GARCH for volatility, (b) Machine learning (XGBoost, LSTM) for non-linear patterns, (c) Hybrid: ARIMA + ML ensemble. (5) Feature limitations: Returns alone insufficient. Need: volume, volatility, sentiment. Solution: ARIMAX (ARIMA with exogenous variables): Y_t = ARIMA + β*Volume_t + γ*VIX_t. (6) Forecast horizon: ARIMA degrades beyond 5-10 steps. 1-day forecast OK, 20-day forecast → mean. Check: Plot forecast error vs horizon. Solution: Use ARIMA only for short-term (1-5 days), long-term use trend models. Implementation: Build ensemble: 40% ARIMA (short-term), 30% GARCH (volatility), 30% XGBoost (non-linear). Combine forecasts with weighted average. Re-train weekly.',
    keyPoints: [
      'Overfitting: high in-sample, low out-of-sample R². Use BIC, limit p,q ≤ 3',
      'Parameter instability: rolling coefficients vary. Use shorter windows (126 days)',
      'Regime changes: Chow test for breaks. Use regime-switching or separate models',
      'Non-linearity: errors during extremes. Add GARCH, ML ensemble',
      'Limited features: add exogenous variables (ARIMAX), combine multiple models',
    ],
  },
];
