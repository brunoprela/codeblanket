export const kalmanFiltersQuiz = [
  {
    id: 1,
    question:
      'Compare three methods for estimating time-varying beta in a pairs trading strategy: (A) Rolling 60-day OLS regression, (B) EWMA (Exponentially Weighted Moving Average) with λ=0.94, (C) Kalman filter. For a strategy with daily rebalancing: Which method adapts fastest to regime changes? Which is most stable? Which performs best out-of-sample? Design an experiment comparing all three across 2008 financial crisis, 2020 COVID crash, and normal periods.',
    answer: `[Comprehensive outline: Rolling OLS: slow adaptation (60-day lag), stable but outdated; EWMA: moderate adaptation (effective window ~17 days), balance of speed/stability; Kalman: fastest adaptation with optimal noise filtering, but can overfit if misspecified; Performance: Kalman wins during regime changes (2008, 2020), EWMA best in normal periods, rolling OLS most stable but lags; Experiment design: 10-year backtest with regime labels, compare by Sharpe ratio and drawdown; Implementation with code showing all three methods side-by-side.]`,
  },
  {
    id: 2,
    question:
      "A quantitative researcher proposes using Kalman filter to 'predict' stock returns by modeling returns as hidden state with price as noisy observation. The backtest shows impressive results (Sharpe 2.5). Critique this approach: (1) Why is this model specification fundamentally flawed? (2) What is the Kalman filter actually doing? (3) Why do backtests look good but live trading will fail? (4) Correct applications of Kalman filters in finance.",
    answer: `[Critical analysis: Returns are NOT hidden states (they're observed!), Kalman assumes return is constant with white noise → just smoothing, backtests overfit to noise, forward-looking bias in parameter tuning; What it's really doing: Sophisticated moving average creating lag; Live trading fails: smoothing introduces lag, parameters change; Correct uses: hidden variables (beta, volatility, spread mean), not returns themselves. Code demonstrating the flaw.]`,
  },
  {
    id: 3,
    question:
      'Design a Kalman filter for estimating time-varying volatility σₜ from observed returns. Compare to GARCH(1,1). Under what conditions would Kalman be preferred? Address: state-space formulation, parameter selection (process/observation noise), computational efficiency, and forecasting performance.',
    answer: `[Answer outline: State equation: ln(σₜ²) = ln(σₜ₋₁²) + wₜ; Observation: rₜ² ≈ σₜ² + vₜ; Kalman preferred when: (1) need real-time updating, (2) want probabilistic intervals, (3) have irregular data; GARCH better for: (1) leverage effects, (2) established financial theory, (3) longer-term forecasts; Computational: Kalman O(1) per update, GARCH requires re-estimation; Both produce similar volatility estimates in practice. Implementation comparing both.]`,
  },
];
