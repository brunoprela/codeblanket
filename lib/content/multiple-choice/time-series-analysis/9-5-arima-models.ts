export const arimaModelsMultipleChoice = [
  {
    id: 1,
    question:
      'A time series is tested with the ADF test and yields p-value = 0.42. After first differencing, the ADF p-value = 0.01. What does this tell you about the appropriate ARIMA model?',
    options: [
      'd=0 (no differencing needed), use ARMA model',
      'd=1 (one difference), series is I(1)',
      'd=2 (two differences), series is I(2)',
      'Cannot determine without knowing p and q',
      'Data is non-stationary even after differencing',
    ],
    correctAnswer: 1,
    explanation:
      "d=1 is appropriate. Original series: ADF p=0.42 > 0.05 → fail to reject unit root → non-stationary. After first difference: ADF p=0.01 < 0.05 → reject unit root → stationary. This indicates the series is I(1) (integrated of order 1). Definition: A series is I(d) if it becomes stationary after d differences. For ARIMA(p,d,q), use d=1. Then determine p and q using ACF/PACF of the differenced series. Common in finance: stock prices are I(1), returns are I(0). Warning: d≥2 very rare in finance. If first difference isn't stationary, check for: structural breaks, seasonality, or data issues before trying d=2.",
    difficulty: 'beginner',
  },
  {
    id: 2,
    question:
      'An ARIMA(0,1,1) model is fit to daily stock prices with θ = -0.2. What is the 1-step ahead forecast for tomorrow's price if today's price is $100 and today's residual is $0.50?',
    options: [
      '$100.00 (random walk forecast)',
      '$99.90 (incorporating negative MA term)',
      '$100.10 (incorporating positive residual)',
      '$99.50 (price minus residual)',
      '$100.50 (price plus residual)',
    ],
    correctAnswer: 1,
    explanation:
      "Forecast is $99.90. ARIMA(0,1,1) model: $P_t - P_{t-1} = ε_t + θε_{t-1}$. Rearranged: $P_t = P_{t-1} + ε_t + θε_{t-1}$. For 1-step forecast: $\\hat{P}_{t+1} = P_t + \\mathbb{E}[ε_{t+1}] + θε_t$. Since $\\mathbb{E}[ε_{t+1}] = 0$ (future error unknown): $\\hat{P}_{t+1} = P_t + θε_t = 100 + (-0.2)(0.50) = 100 - 0.10 = 99.90$. Interpretation: Negative θ (common for stock prices) creates mean reversion. Today's positive residual (+$0.50, price above model expectation) pulls tomorrow's forecast DOWN. For θ=-0.2, 20% of today's shock reverses tomorrow. If θ=0: random walk (forecast = $100). MA term captures short-term overreaction correction.",
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question:
      'You fit SARIMA(1,1,1)(1,1,1)₁₂ to monthly sales data. How many parameters does this model have (excluding the constant)?',
    options: [
      '3 parameters (1 AR + 1 MA + 1 seasonal)',
      '6 parameters (p+q+P+Q+d+D)',
      '4 parameters (1 AR + 1 MA + 1 seasonal AR + 1 seasonal MA)',
      '8 parameters (include differencing and seasonal period)',
      '2 parameters (only AR and MA)',
    ],
    correctAnswer: 2,
    explanation:
      "4 parameters total (excluding constant). SARIMA(p,d,q)(P,D,Q)ₛ parameters: (1) Non-seasonal AR: φ₁, ..., φₚ → 1 parameter (p=1), (2) Non-seasonal MA: θ₁, ..., θ_q → 1 parameter (q=1), (3) Seasonal AR: Φ₁, ..., Φₚ → 1 parameter (P=1), (4) Seasonal MA: Θ₁, ..., Θ_Q → 1 parameter (Q=1). The d and D are differencing orders (not parameters), s=12 is the seasonal period (fixed). Total parameters to estimate: p + q + P + Q = 1+1+1+1 = 4. If include constant: 5 parameters. Warning: High parameter count risks overfitting. For 5 years monthly data (60 obs), 4 parameters is ~7% of data → reasonable. Seasonal models need sufficient data: want at least 4-5 full seasonal cycles (48-60 months for monthly with s=12).",
    difficulty: 'intermediate',
  },
  {
    id: 4,
    question:
      'An ARIMA(1,1,0) model with φ=0.8 is used to forecast stock prices. What is the implied long-term growth rate if the drift term is c=0.5?',
    options: [
      '0.5 per period (the drift term)',
      '2.5 per period (c/(1-φ) = 0.5/0.2)',
      '0.4 per period (c × φ)',
      'Undefined - random walk has no long-term growth',
      '0.1 per period (c - φ)',
    ],
    correctAnswer: 0,
    explanation:
      "Long-term growth is 0.5 per period (the drift term). ARIMA(1,1,0) with drift: $\\Delta P_t = c + φ \\Delta P_{t-1} + ε_t$. This is AR(1) on DIFFERENCES (returns), not levels. As h→∞: $\\mathbb{E}[\\Delta P_{t+h}] \\rightarrow c/(1-φ)$ for AR(1) on stationary series... BUT we're in levels! Expanding: $P_{t+h} = P_t + \\sum_{i=1}^h \\Delta P_{t+i}$. Expected h-period change: $\\mathbb{E}[P_{t+h} - P_t] = h \\cdot c$ (drift accumulates linearly!). So: Long-term growth = c per period. The φ affects short-term dynamics (momentum in returns) but not long-term drift. Common mistake: applying AR(1) formula to integrated series. Remember: I(1) series have deterministic trend = drift. If c=0: random walk (no drift). For stock prices, c≈0 (weak drift), so approximately random walk.",
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'After fitting an ARIMA model, the residuals show significant ACF at lag 12 (p-value < 0.01) but no other lags are significant. The data is monthly. What action should you take?',
    options: [
      'Increase AR order to capture lag 12',
      'Increase MA order to capture lag 12',
      'Add seasonal component with period 12',
      'Model is adequate, ignore single significant lag',
      'Take another difference (increase d)',
    ],
    correctAnswer: 2,
    explanation:
      "Add seasonal component (SARIMA with s=12). Residual ACF significant at lag 12 (but not nearby lags) suggests ANNUAL seasonality: lag 12 = 12 months = 1 year. This pattern indicates your non-seasonal ARIMA missed seasonal dynamics. Solution: Upgrade to SARIMA(p,d,q)(P,D,Q)₁₂. Start with (P,D,Q) = (1,0,1) or (0,1,1) at seasonal frequency. Why not high-order AR/MA? Adding AR(12) or MA(12) requires 12+ parameters (wasteful). Seasonal models are parsimonious: Φ₁₂ captures annual pattern with 1 parameter. Example: If current model is ARIMA(1,1,1), try SARIMA(1,1,1)(1,0,1)₁₂ or SARIMA(1,1,1)(0,1,1)₁₂. Re-fit and check if lag 12 ACF becomes insignificant. Common in finance: Quarterly earnings → lag 12 in monthly stock data. Retail sales → December spike. Always inspect ACF for seasonal patterns!",
    difficulty: 'advanced',
  },
];

