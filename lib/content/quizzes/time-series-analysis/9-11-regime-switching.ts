export const regimeSwitchingQuiz = [
  {
    id: 1,
    question:
      'Design a trading system that dynamically adjusts leverage based on detected volatility regime (low/medium/high). Address: regime detection methodology (real-time vs lagged), position sizing rules, regime transition management, performance during 2008/2020 crises, and avoiding over-optimization.',
    answer: `[Outline: Use Markov-switching with 3 regimes based on VIX levels; Real-time detection via filtered probabilities (not smoothed - look-ahead bias); Leverage: 3x in low vol, 1x medium, 0.5x high; Transitions: gradual adjustment over 5 days to avoid whipsaw; Backtests show ~30% drawdown reduction vs fixed leverage; Avoid overfitting: out-of-sample validation, regime-independent rules, minimum hold periods.]`,
  },
  {
    id: 2,
    question:
      'A quantitative researcher finds that ARIMA forecast errors are much larger during certain periods (2008, 2020). Propose a regime-switching ARIMA framework that: detects crisis regimes automatically, uses different ARIMA parameters per regime, and provides regime-aware forecast intervals. Compare to single-regime ARIMA.',
    answer: `[Framework: Markov-switching ARIMA where parameters (φ, θ, σ) switch between regimes; Crisis detection via transition probabilities + external indicators (VIX > 30); Regime-specific parameters capture different dynamics (crisis: higher persistence, different volatility); Forecast intervals widen in crisis regime → better calibration; Performance: MSE 20-30% lower in regime-switching vs single regime; Cost: more parameters, overfitting risk, complexity.]`,
  },
  {
    id: 3,
    question:
      'Compare three regime detection methods for portfolio management: (1) Markov-switching (model-based), (2) Machine learning clustering of market features, (3) Rule-based (e.g., VIX > 25 = crisis). For a $1B hedge fund: Which is most robust? Most interpretable? Best real-time performance? Design hybrid approach combining all three.',
    answer: `[Markov-switching: Probabilistic, smooth transitions, but black-box and lag in detection; ML clustering: Flexible, many features (vol, correlations, momentum), but unstable and hard to interpret; Rule-based: Transparent, fast, but arbitrary thresholds and false signals; Hybrid: Use rules for preliminary signal, Markov for probabilistic regime estimate, ML for regime characterization; Weight by historical accuracy; Robust fund system uses ensemble with confidence-weighted decisions.]`,
  },
];
