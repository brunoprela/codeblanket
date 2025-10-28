export const autocorrelationPartialAutocorrelationMultipleChoice = [
  {
    id: 1,
    question:
      'A time series shows ACF that decays slowly (0.9, 0.81, 0.73, 0.66...) and PACF with a large spike at lag 1 (0.9) then values near zero at higher lags. What is the correct model identification?',
    options: [
      'MA(1) - ACF cuts off after lag 1',
      'AR(1) with φ = 0.9 - PACF cuts off after lag 1',
      'ARMA(1,1) - both ACF and PACF decay',
      'Non-stationary - slow ACF decay indicates unit root',
      'White noise - ACF values are within confidence bands',
    ],
    correctAnswer: 1,
    explanation:
      'This is a textbook AR(1) signature: (1) PACF cuts off sharply after lag 1 → AR order = 1, (2) ACF decays geometrically (0.9, 0.81=0.9², 0.73≈0.9³) → confirms AR, (3) PACF(1)=0.9 is the AR coefficient φ. The geometric decay rate matches φ^k. NOT non-stationary because: φ=0.9 < 1 (stationary), and decay is consistent geometric (not slow random walk decay). NOT MA because MA(1) would have ACF that cuts off after lag 1, not geometric decay. Key distinction: AR → PACF cuts off, ACF decays. MA → ACF cuts off, PACF decays.',
  },
  {
    id: 2,
    question:
      'After fitting an AR(1) model to stock returns, the Ljung-Box test on residuals gives p-value = 0.42 (cannot reject white noise), but the Ljung-Box test on squared residuals gives p-value = 0.001 (reject white noise). What does this indicate?',
    options: [
      'The AR(1) model is adequate; squared residuals are always correlated',
      'Volatility clustering (GARCH effects) - need to model time-varying variance',
      'The AR(1) order is too low; need AR(2) or higher',
      'Data quality issue - squared residuals should match regular residuals',
      'This is normal and can be ignored',
    ],
    correctAnswer: 1,
    explanation:
      "This is the SIGNATURE of volatility clustering that requires GARCH modeling: (1) Residuals white noise (p=0.42 > 0.05) → Mean equation (AR part) is adequate. (2) Squared residuals correlated (p=0.001 < 0.05) → Variance is TIME-VARYING and PREDICTABLE. This means: Returns = μ + ε_t where ε_t = σ_t × z_t. While z_t is unpredictable, σ_t is predictable from past squared shocks. Solution: AR(1)-GARCH(1,1) model. Why not other options: (A) Wrong - squared residuals should be white noise if variance is constant, (C) Wrong - adding AR lags won't fix variance clustering, (D) Wrong - this is a model specification issue, not data problem, (E) VERY wrong - ignoring this means underestimating risk and getting wrong confidence intervals!",
  },
  {
    id: 3,
    question:
      'A pairs trading spread has ACF(1) = 0.95. The half-life of mean reversion is approximately how many periods, and what does this imply for trading?',
    options: [
      'Half-life ≈ 1 period; fast mean reversion, ideal for daily trading',
      'Half-life ≈ 7 periods; moderate mean reversion, suitable for weekly trading',
      'Half-life ≈ 14 periods; slow mean reversion, requires wide stops and long holding periods',
      'Half-life ≈ 0.5 periods; very fast mean reversion, trade at any deviation',
      'Half-life is infinite; spread does not mean-revert',
    ],
    correctAnswer: 2,
    explanation:
      'Half-life formula: t_½ = -ln(2)/ln(φ) where φ = ACF(1). For φ=0.95: t_½ = -ln(2)/ln(0.95) = -0.693/(-0.0513) ≈ 13.5 periods. This indicates VERY SLOW mean reversion. Trading implications: (1) Long holding periods (13.5 periods to decay 50%, ~45 periods to decay 90%), (2) Need wide entry thresholds (2.5-3σ instead of 2σ), (3) Wide stop-losses (spread can overshoot significantly), (4) Low trade frequency, (5) Higher capital requirements (capital tied up longer). Compare: φ=0.5 → half-life = 1 period (fast), φ=0.7 → half-life = 2.4 periods (moderate). High ACF ≈ 1 is BAD for mean-reversion trading (not good!) because convergence is slow and risky. Infinite half-life only if φ ≥ 1 (unit root).',
  },
  {
    id: 4,
    question:
      'The ACF of daily S&P 500 returns shows all values within confidence bands except for a spike at lag 5 (ACF(5) = -0.12, significant). What is the MOST likely explanation?',
    options: [
      'Returns follow MA(5) model - can be exploited for profit',
      'Data error or microstructure noise - returns should be white noise under EMH',
      'Friday-to-Monday effect or calendar pattern worth investigating',
      'Spurious result from multiple testing - 1 in 20 lags significant by chance',
      'The market has a 5-day cycle that can be traded',
    ],
    correctAnswer: 3,
    explanation:
      "With multiple hypothesis testing (testing 20+ lags), we expect ~1 false positive at α=0.05 level. A single significant lag out of many is likely SPURIOUS. Statistical evidence: (1) If testing 20 lags at 5% significance, expect 20 × 0.05 = 1 false positive by chance, (2) One significant lag doesn't form coherent pattern, (3) Bonferroni correction: adjusted α = 0.05/20 = 0.0025, likely not significant after correction. Proper validation: (1) Check if lag-5 significance persists in different sub-periods, (2) Out-of-sample validation, (3) Check if economic explanation exists (e.g., settlement lags), (4) Test if pattern exists in other markets. Why NOT other options: (A) MA(5) would show significance at multiple lags 1-5, (C) Possible but need more evidence than single lag, (E) Cycles don't create single-lag spikes. Bottom line: Don't trade based on single significant lag - likely data mining artifact.",
  },
  {
    id: 5,
    question:
      'Your ACF/PACF analysis suggests an ARMA(2,2) model for a time series. However, an ARMA(1,1) model with lower AIC and BIC also passes all diagnostic tests (residuals are white noise). Which model should you use in production?',
    options: [
      'ARMA(2,2) - matched the ACF/PACF pattern exactly',
      'ARMA(1,1) - simpler model with better information criteria and adequate fit',
      'Average predictions from both models for robustness',
      'ARMA(2,2) - more parameters mean better forecast accuracy',
      'Neither - if two models work, data quality is suspect',
    ],
    correctAnswer: 1,
    explanation:
      "Choose ARMA(1,1) based on PARSIMONY PRINCIPLE (Occam's Razor): When multiple models fit adequately, choose the simplest. Evidence: (1) Lower AIC/BIC → ARMA(1,1) preferred by information criteria (penalize complexity), (2) Residuals white noise → Model is adequate (captures all patterns), (3) Fewer parameters → Less overfitting risk, (4) Better out-of-sample performance (typically), (5) Easier to interpret and maintain. Box-Jenkins methodology: ACF/PACF guide initial selection, but information criteria make final choice. ARMA(2,2) risks: (1) Overfitting - capturing noise as signal, (2) Parameter estimation uncertainty increases with more parameters, (3) Out-of-sample degradation. Why NOT average: (A) Model averaging adds complexity without benefit if both adequate, (B) ARMA(1,1) sufficient. Rule: If diagnostics pass, prefer simpler model. Only use complex model if it significantly improves fit (lower IC + better diagnostics).",
  },
];
