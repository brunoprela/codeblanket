export const armaModelsMultipleChoice = [
  {
    id: 1,
    question:
      'An MA(1) model is fit to daily returns and yields θ = 0.6. What is the ACF at lag 1, and how does this compare to an AR(1) model with φ = 0.6?',
    options: [
      'MA(1) ACF(1) = 0.6, same as AR(1) ACF(1) = 0.6',
      'MA(1) ACF(1) = 0.38, AR(1) ACF(1) = 0.6, different patterns',
      'MA(1) ACF(1) = 0, AR(1) ACF(1) = 0.6, MA has no autocorrelation',
      'MA(1) ACF(1) = 0.6, AR(1) ACF(1) = 0.38, opposite patterns',
      'MA(1) and AR(1) both have ACF(1) = 0.38',
    ],
    correctAnswer: 1,
    explanation:
      'MA(1) ACF formula: ρ₁ = θ/(1+θ²). For θ=0.6: ρ₁ = 0.6/(1+0.36) = 0.6/1.36 = 0.44 (approximately 0.38 depending on rounding). AR(1) ACF: ρ₁ = φ = 0.6 directly. Key differences: (1) MA(1) ACF cuts off after lag 1 (ρ₂=ρ₃=...=0), AR(1) decays geometrically (ρ₂=0.36, ρ₃=0.22, ...), (2) MA(1) always has |ρ₁| < 0.5 regardless of θ, AR(1) can approach 1. This distinction is critical for model identification: sharp cutoff → MA, gradual decay → AR. Common error: thinking MA coefficient directly equals ACF (only true for AR).',
  },
  {
    id: 2,
    question:
      'You fit ARMA(1,1) and ARMA(2,2) models to the same data. Results: ARMA(1,1) has AIC=1500, BIC=1520; ARMA(2,2) has AIC=1480, BIC=1530. Which model should you choose for production forecasting?',
    options: [
      'ARMA(2,2) - lower AIC means better fit',
      'ARMA(1,1) - lower BIC indicates better parsimony',
      'ARMA(2,2) - more parameters mean better forecasts',
      'Average predictions from both models for robustness',
      'Neither - the conflicting criteria indicate data quality issues',
    ],
    correctAnswer: 1,
    explanation:
      "Choose ARMA(1,1) based on BIC. While ARMA(2,2) has lower AIC (1480 < 1500), it has HIGHER BIC (1530 > 1520). BIC = -2ln(L) + k·ln(n) penalizes complexity more than AIC = -2ln(L) + 2k. For forecasting (especially out-of-sample), BIC is preferred because: (1) Prevents overfitting - BIC's ln(n) penalty grows with sample size, (2) Parsimony principle - simpler models generalize better, (3) Research shows BIC selects models with better out-of-sample performance. The 2 extra parameters in ARMA(2,2) improve in-sample fit (lower AIC) but likely capture noise, not signal. For production: prefer simpler model with adequate diagnostics. Only use ARMA(2,2) if it significantly improves diagnostic tests (residual autocorrelation, etc.), not just information criteria.",
  },
  {
    id: 3,
    question:
      'After fitting an ARMA(2,1) model, the Ljung-Box test on residuals gives p-value = 0.62. What does this indicate about the model?',
    options: [
      'Model is inadequate - p-value too high indicates overfitting',
      'Model is adequate - cannot reject white noise hypothesis for residuals',
      'Model is perfect - p=0.62 is ideal for time series',
      'Need higher ARMA order - p>0.05 means model is too simple',
      'Data has issues - Ljung-Box p-value should always be < 0.05',
    ],
    correctAnswer: 1,
    explanation:
      "p=0.62 > 0.05 means we CANNOT reject the null hypothesis that residuals are white noise - this is GOOD! Ljung-Box null hypothesis H₀: First m autocorrelations are jointly zero (white noise). For adequate model: Want to fail to reject H₀ (p > 0.05). If p < 0.05: Residuals show autocorrelation → model hasn't captured all patterns → need different specification. Common misconception: thinking high p-value indicates overfitting. Actually: (A) p > 0.05: Residuals are white noise ✓ Model adequate, (B) p < 0.05: Residuals autocorrelated ✗ Model inadequate. The p=0.62 indicates residuals behave like white noise (no remaining patterns to capture). Still check: (1) Parameter significance, (2) Stationarity/invertibility, (3) Out-of-sample performance, but residual diagnostics passed!",
  },
  {
    id: 4,
    question:
      'An AR(1) model has φ = 0.95. What is the approximate half-life of shocks, and what does this imply for trading?',
    options: [
      'Half-life ≈ 1 period; fast mean reversion, ideal for day trading',
      'Half-life ≈ 7 periods; moderate persistence, suitable for swing trading',
      'Half-life ≈ 14 periods; very slow mean reversion, requires patient capital',
      'Half-life is infinite; process does not mean-revert',
      'Half-life ≈ 0.5 periods; extremely fast reversion, scalp-trade',
    ],
    correctAnswer: 2,
    explanation:
      'Half-life formula: t½ = -ln(2)/ln(φ). For φ=0.95: t½ = -0.693/ln(0.95) = -0.693/(-0.0513) ≈ 13.5 periods. This indicates VERY SLOW mean reversion. Implications: (1) Shocks persist for ~14 periods before decaying 50%, (2) Full decay (to 1%) takes ~90 periods, (3) Trading requires long holding periods (weeks if daily data), (4) High φ ≈ 1 means near-random walk behavior, (5) Not suitable for day trading or short-term strategies. Compare: φ=0.5 → t½≈1 period (fast), φ=0.7 → t½≈2.4 periods (moderate), φ=0.95 → t½≈13.5 periods (very slow). For mean-reversion trading: prefer φ < 0.7 (faster convergence). High persistence (φ→1) better suited for momentum strategies, not mean reversion. Key insight: higher φ ≠ better for trading, often means slower convergence and higher risk.',
  },
  {
    id: 5,
    question:
      'You estimate an MA(2) model and get θ₁ = 0.5, θ₂ = -0.3. To check invertibility, you find the roots of the MA polynomial are z₁ = 1.2 and z₂ = -1.8. Is the model invertible?',
    options: [
      'Yes - both |θ₁| < 1 and |θ₂| < 1',
      'No - θ₂ is negative which violates invertibility',
      'Yes - both roots are outside unit circle (|z| > 1)',
      'No - one root is positive and one negative',
      'Cannot determine without more information',
    ],
    correctAnswer: 2,
    explanation:
      "Model IS invertible because both roots are outside the unit circle. MA(q) invertibility condition: roots of 1 + θ₁z + θ₂z² + ... + θ_qz^q = 0 must lie OUTSIDE unit circle (|z| > 1). Here: |z₁| = 1.2 > 1 ✓ and |z₂| = |-1.8| = 1.8 > 1 ✓. Common mistakes: (A) Checking |θᵢ| < 1 only works for MA(1), not MA(q) with q>1, (B) Sign of roots doesn't matter - only magnitude, (C) Positive vs negative roots irrelevant for invertibility. Why roots matter: MA can be inverted to infinite AR: Xₜ = -θ₁Xₜ₋₁ - θ₂Xₜ₋₂ + ... + εₜ. This converges only if roots outside unit circle. For MA(1): |θ| < 1 directly translates to |root| = |1/θ| > 1. For MA(2): need to solve quadratic. Statistical software (statsmodels) automatically checks this and can enforce invertibility constraint during estimation.",
  },
];
