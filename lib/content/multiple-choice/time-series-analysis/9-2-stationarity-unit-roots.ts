export const stationarityUnitRootsMultipleChoice = [
  {
    id: 1,
    question:
      'You run an ADF test on daily stock prices and get p-value = 0.42. Then you run the same test on daily returns and get p-value = 0.0001. What is the correct interpretation?',
    options: [
      'Both tests show the data is non-stationary; cannot use for modeling',
      'Prices have unit root (non-stationary) but returns are stationary; use returns for modeling',
      'The tests contradict each other; data quality issue suspected',
      'Prices are stationary but returns are not; difference the returns',
      'P-value > 0.05 means stationary, so prices are fine to use',
    ],
    correctAnswer: 1,
    explanation:
      "ADF null hypothesis is 'unit root exists'. For prices: p=0.42 > 0.05 → fail to reject null → prices have unit root (non-stationary). For returns: p=0.0001 < 0.05 → reject null → returns are stationary. This is the EXPECTED pattern: stock prices follow random walk (non-stationary) but returns are stationary. Always model returns, not prices. This is not a contradiction - prices are I(1) and returns are I(0). Common misconception: p-value > 0.05 does NOT mean stationary in ADF (opposite!).",
  },
  {
    id: 2,
    question:
      'Two stocks are being evaluated for pairs trading. Stock A: ADF p-value = 0.50 (levels), 0.001 (1st diff). Stock B: ADF p-value = 0.48 (levels), 0.002 (1st diff). Cointegration test on levels: p-value = 0.03. What should you conclude?',
    options: [
      'Not suitable for pairs trading because both stocks are non-stationary',
      'Difference both stocks first, then test cointegration on the differenced series',
      'EXCELLENT pairs trading opportunity: both I(1) and cointegrated',
      'Test failed because cointegration requires stationary series',
      'Need to difference to I(0) before calculating cointegration',
    ],
    correctAnswer: 2,
    explanation:
      'Perfect pairs trading setup! Both stocks are I(1): non-stationary in levels (p>0.05) but stationary in first differences (p<0.05). The cointegration test p=0.03 < 0.05 means their linear combination IS stationary despite both being non-stationary individually. This is the DEFINITION of cointegration. Common error: thinking you need to difference first - NO! Cointegration REQUIRES non-stationary series. If you difference first, you destroy the cointegrating relationship. The mean-reverting spread (B - β*A) is your trading signal. Key insight: I(1) + I(1) normally = I(2), but if cointegrated, their combination is I(0). This special property is what makes pairs trading profitable.',
  },
  {
    id: 3,
    question:
      'A time series shows: ADF test p-value = 0.04 (reject unit root), but KPSS test p-value = 0.03 (reject stationarity). The tests disagree. What is the MOST likely explanation?',
    options: [
      'Data quality error - one test must be wrong, discard the data',
      'The series has structural break or regime change that confuses both tests',
      'Tests are measuring different things and always disagree; ignore KPSS',
      'The series is exactly on the boundary (φ = 1.0) and both tests are correct',
      'ADF is more reliable; conclude series is stationary',
    ],
    correctAnswer: 1,
    explanation:
      "Conflicting ADF/KPSS results (ADF says stationary, KPSS says non-stationary) typically indicate STRUCTURAL BREAK. ADF tests for unit root assuming constant parameters. KPSS tests for constant mean. If there's a regime change (mean shifts from μ₁ to μ₂), ADF might not detect the unit root, but KPSS will reject constant mean. Example: Returns were volatile 2019-2020 (low mean), then stable 2021-2022 (high mean). Each regime is stationary, but KPSS sees overall non-stationarity. Solution: (1) Test each regime separately, (2) Use regime-switching model, (3) Run Chow test for breaks. NOT a data error - this is informative! Don't discard data, investigate the regime change. This is common in financial data during crisis periods.",
  },
  {
    id: 4,
    question:
      'You difference a non-stationary series once and it becomes stationary (ADF p < 0.01). Your colleague suggests differencing again "to be extra sure it is stationary." What is wrong with this approach?',
    options: [
      'Nothing wrong; more differencing always improves stationarity',
      'Over-differencing introduces artificial negative autocorrelation and makes forecasting worse',
      'Second differencing will make it non-stationary again (oscillates)',
      'You can only difference a maximum of once by mathematical definition',
      'The series needs to be tested for seasonality before second differencing',
    ],
    correctAnswer: 1,
    explanation:
      "OVER-DIFFERENCING is a serious error! If a series is I(1) (stationary after 1st difference), differencing again creates I(0) - I(0) = I(-1), which introduces ARTIFICIAL negative autocorrelation. Example: Let X_t be stationary (mean-reverting). Then ΔX_t = X_t - X_{t-1} has negative autocorrelation: if X_t was above mean, X_{t-1} likely below mean (mean reversion), so difference is positive followed by negative. This looks like a pattern but it's an artifact! Over-differenced series: (1) Has negative ACF at lag 1, (2) Worse forecast performance (added noise), (3) Harder to interpret economically. Rule: Difference minimally. If ADF confirms stationary after 1 difference, STOP. Most financial series are I(1) - prices need 1 difference, returns need 0. Second differencing is rarely needed (only for I(2) series like inflation of inflation).",
  },
  {
    id: 5,
    question:
      'An ML model trained on raw stock prices (not returns) achieves R² = 0.95 in-sample but R² = 0.10 out-of-sample. An ML model trained on returns achieves R² = 0.08 in-sample and R² = 0.06 out-of-sample. Which model is better and why?',
    options: [
      'Price model (R²=0.95 is much better than 0.08)',
      'Return model (stable performance indicates real predictive power)',
      'Price model (higher R² means better predictions)',
      'They are equivalent since out-of-sample R² is similar (0.10 vs 0.06)',
      'Both are poor models (R² < 0.10 out-of-sample)',
    ],
    correctAnswer: 1,
    explanation:
      "Return model is FAR superior despite lower R²! The price model's performance is SPURIOUS: (1) High in-sample R² (0.95) comes from autocorrelation in non-stationary prices (trend), not real predictive power. Random walk achieves R² > 0.90 with naive forecast! (2) Massive degradation (0.95 → 0.10) indicates overfitting to in-sample trend. (3) Out-of-sample R²=0.10 is barely better than random walk. Return model characteristics: (1) Low but HONEST in-sample R² (0.08) reflects true difficulty of predicting returns. (2) Minimal degradation (0.08 → 0.06) shows stable, generalizable patterns. (3) Model hasn't overfit to spurious trends. Reality check: Financial returns are HARD to predict - R²=0.06 that's consistent is more valuable than R²=0.95 that's spurious. For trading: 53% direction accuracy (from R²=0.06) can be profitable if consistent. Focus on out-of-sample Sharpe ratio and economic significance, not in-sample R².",
  },
];
