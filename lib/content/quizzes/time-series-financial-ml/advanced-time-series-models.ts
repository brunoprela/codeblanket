export const advancedTimeSeriesModelsQuiz = [
  {
    id: 'atsm-q-1',
    question:
      'You are building a risk management system that requires accurate 1-day and 10-day ahead volatility forecasts for position sizing. Design a complete GARCH-based system: (1) model selection (GARCH vs GJR-GARCH vs EGARCH), (2) parameter estimation and validation, (3) forecast generation, (4) converting volatility forecasts to position sizes. How do you handle the leverage effect? Why is volatility more predictable than returns?',
    sampleAnswer:
      'Complete GARCH system: (1) Model selection: Test for leverage effect by checking asymmetry in volatility response. Calculate correlation (return_t-1, volatility_t). If significantly negative (< -0.1), use GJR-GARCH or EGARCH. For SPY, typically find γ ≈ 0.1-0.2 in GJR-GARCH, confirming leverage effect. Compare models with AIC: GJR-GARCH usually wins for equity indices. (2) Parameter estimation: Fit GARCH(1,1): σ_t² = ω + α*ε² + β*σ². Check persistence: α + β < 1 (stationary). Typical values: ω ≈ 0.01, α ≈ 0.05-0.10, β ≈ 0.85-0.90. Persistence α + β ≈ 0.95 (high). Half-life = -log(2)/log(0.95) ≈ 14 days. Validate: Plot standardized residuals (r_t/σ_t), should have constant variance. Run Ljung-Box test on squared standardized residuals (should show no autocorrelation). (3) Forecasting: 1-day: σ²_t+1 = ω + α*ε²_t + β*σ²_t. 10-day: Use recursive formula or closed-form for long-horizon. σ²_t+h converges to long-run variance: ω/(1-α-β). For h=10, approximately: σ²_t+10 ≈ 0.7*current + 0.3*long_run. (4) Position sizing: Target portfolio volatility (e.g., 15% annual). Calculate position as: position = (target_vol / forecasted_vol) * capital. If forecasted vol = 1% daily → 15.8% annual → position = (15/15.8) * 100k = $94.9k. If vol spikes to 3% daily → 47.4% annual → position = (15/47.4) * 100k = $31.6k. Why volatility predictable: Returns show minimal autocorrelation (ACF ≈ 0), but squared returns (volatility proxy) show high autocorrelation (ACF ≈ 0.2-0.4 for lags 1-20). This is volatility clustering: high vol today → high vol tomorrow. Returns are near-random walk, but volatility follows predictable patterns. Leverage effect: Negative returns → increased volatility because: (1) leverage ratio increases (debt/equity), (2) investor uncertainty rises. Result: Need asymmetric models (GJR-GARCH) to capture this.',
    keyPoints: [
      'Test for leverage effect: corr (return_t-1, vol_t) < 0 → use GJR-GARCH',
      'Validate persistence: α + β < 1, typically ≈ 0.95 (high persistence, 14-day half-life)',
      'Multi-horizon forecast: short-term uses recent vol, long-term converges to long-run avg',
      'Position sizing: position = (target_vol / forecast_vol) * capital (inverse relationship)',
      'Volatility predictable due to clustering; returns unpredictable due to efficiency',
    ],
  },
  {
    id: 'atsm-q-2',
    question:
      'Design a multi-asset trading system using VAR to model dependencies between stocks (SPY), bonds (TLT), and gold (GLD). How do you: (1) test for Granger causality, (2) generate impulse response functions, (3) use these insights for portfolio construction and hedging? Explain how shocks propagate across assets and implications for diversification.',
    sampleAnswer:
      "VAR multi-asset system: (1) Granger causality testing: Tests if past values of X help predict Y beyond Y's own history. For SPY → TLT: Estimate restricted model (TLT depends only on lagged TLT), unrestricted model (TLT depends on lagged TLT + lagged SPY), F-test on difference. If p < 0.05, SPY Granger-causes TLT. Example results: SPY → TLT (p = 0.01, significant): Stock market predicts bond moves (flight-to-safety). TLT → SPY (p = 0.23, not significant): Bonds don't predict stocks. GLD → SPY (p = 0.15): Weak gold-to-stock causality. Interpretation: Stock volatility drives portfolio rebalancing into bonds/gold. (2) Impulse Response Functions: Measures dynamic effect of 1% shock to asset i on asset j over time. Example: 1% negative shock to SPY → TLT increases 0.3% same day (flight-to-safety), peaks at 0.5% after 3 days, decays to 0 by day 10. GLD increases 0.2% by day 5 (safe haven demand). SPY shock affects TLT significantly, but TLT shock barely affects SPY (asymmetric relationship). (3) Portfolio construction: Use correlation matrix and IRF for hedging ratios. If 1% SPY shock → 0.5% TLT response, hedge ratio = 0.5. To hedge $100k SPY exposure: Short $50k TLT. Dynamic hedging: Update VAR every month as correlations change. Crisis periods: SPY-TLT correlation becomes more negative (better hedge). Normal periods: Weaker relationship. (4) Diversification: IRF shows shock propagation speed. Fast propagation (1-2 days) → limited diversification during crashes. Slow propagation (10+ days) → time to rebalance. SPY→TLT peaks day 3 → rebalancing lag. For high-frequency: Can't diversify. For daily: Sufficient time. Key insight: Assets aren't independent. VAR quantifies dependencies. During 2020 crash: All assets fell together (diversification failed short-term), then TLT recovered (hedge worked medium-term).",
    keyPoints: [
      'Granger causality: F-test restricted vs unrestricted model, SPY typically predicts TLT (flight-to-safety)',
      'IRF: 1% SPY shock → 0.3-0.5% TLT response (asymmetric, bonds react to stocks)',
      'Hedge ratio from IRF: shock size determines position sizes (e.g., 0.5 ratio)',
      'Diversification limited short-term: shocks propagate within 1-3 days',
      'Dynamic relationships: Update VAR monthly, crisis vs normal correlations differ',
    ],
  },
  {
    id: 'atsm-q-3',
    question:
      'You are forecasting stock prices using ARIMAX with external variables (volume, volatility, sentiment). How do you: (1) select relevant exogenous variables, (2) avoid lookahead bias in feature engineering, (3) validate that external variables actually improve forecasts beyond plain ARIMA? Include walk-forward testing and comparison metrics. Why might ARIMAX outperform ARIMA?',
    sampleAnswer:
      "ARIMAX implementation: (1) Variable selection: Theory-driven + empirical testing. Candidates: Volume (liquidity proxy), realized volatility (risk), bid-ask spread (microstructure), sentiment (news, Twitter), VIX (market fear), sector ETF returns (momentum). Feature engineering: All features must be calculated with t-1 data to predict t. Volume_t-1 (previous day), rolling_vol_t-1 = std (returns_t-20:t-1). Test each variable individually: Add to ARIMA(1,1,1), compare AIC. If AIC decreases > 2, variable is informative. Typical results: Volume: AIC improvement = 3.2 (informative). Lagged returns: AIC improvement = 1.1 (weak). VIX: AIC improvement = 5.7 (strong). Select top 3-5 variables (avoid overfitting). (2) Avoiding lookahead: Critical rule: Feature at time t uses only data available before t. Bad: Return_t (uses future). Good: Return_t-1 (uses past). Bad: Rolling_mean (t-10:t+10) (uses future). Good: Rolling_mean (t-20:t-1) (uses past). Test: Build features in loop, verify no future data. Example: for t in range(100, len (data)): features_t = calculate_features (data[:t]), target_t = data[t]. (3) Validation: Walk-forward testing (252 days train, 21 days test). Baseline: ARIMA(1,1,1): MAE = 0.82%, RMSE = 1.15%, R² = 0.03. ARIMAX(1,1,1) + Volume + VIX: MAE = 0.68%, RMSE = 0.97%, R² = 0.15. Statistical test: Diebold-Mariano test for forecast accuracy difference. p < 0.05 → ARIMAX significantly better. Why ARIMAX better: Captures information beyond price history. Volume spike → increased volatility tomorrow. VIX spike → mean reversion in stocks. ARIMA assumes price contains all information (EMH). ARIMAX allows market microstructure and sentiment to matter. However: Improvement typically modest (10-20% reduction in MAE). Most price movements still unpredictable. Use ARIMAX for marginal edge in systematic strategies. Diminishing returns: Adding 10+ variables doesn't help (overfitting). Stick to 3-5 theoretically-motivated features.",
    keyPoints: [
      'Variable selection: Theory-driven, test individually (AIC improvement > 2), use top 3-5',
      'Lookahead prevention: Features at t use only data before t, verify in loops',
      'Walk-forward validation: Compare ARIMA vs ARIMAX on MAE/RMSE/R², Diebold-Mariano test',
      'ARIMAX captures microstructure: volume, volatility, sentiment beyond price history',
      'Modest improvement (10-20% MAE reduction), diminishing returns with too many features',
    ],
  },
];
