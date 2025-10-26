export const kalmanFiltersMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'In Kalman filter for pairs trading, increasing the transition covariance (process noise) has what effect?',
    options: [
      'Makes hedge ratio more stable (less responsive)',
      'Makes hedge ratio adapt faster to changes',
      'Has no effect on estimated hedge ratio',
      'Increases observation noise',
      'Causes filter to diverge',
    ],
    correctAnswer: 1,
    explanation:
      "Makes hedge ratio adapt faster. Transition covariance (Q) controls how much the hidden state (hedge ratio) is allowed to change between time steps. Higher Q → Kalman believes state changes a lot → puts more weight on new observations → faster adaptation. Lower Q → Kalman believes state is stable → smooths more → slower adaptation. Observation covariance (R) controls measurement noise → higher R → trusts observations less. Trade-off: High Q adapts fast but noisy, Low Q stable but slow to detect regime changes. Tuning: For pairs trading, typical Q=0.001-0.01 for daily data. Test using cross-validation or maximum likelihood estimation.",
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'Kalman filter vs rolling regression for beta estimation: Which statement is correct?',
    options: [
      'Kalman requires more computation per update',
      'Rolling regression adapts faster to changes',
      'Kalman optimal if beta follows random walk',
      'Rolling regression provides uncertainty estimates',
      'Both give identical results',
    ],
    correctAnswer: 2,
    explanation:
      "Kalman optimal if beta follows random walk. Key insight: Kalman filter is optimal (minimum mean squared error) WHEN model assumptions hold. For beta estimation: If $\\beta_t = \\beta_{t-1} + w_t$ (random walk) → Kalman optimal! Rolling regression: Equal weight to all observations in window → suboptimal if recent data more informative. Computation: Kalman O(1) per update (recursive), Rolling O(window size). Adaptation: Kalman adapts continuously, Rolling abrupt changes at window boundaries. Uncertainty: Kalman provides full covariance, Rolling only point estimates (need bootstrap). When Kalman wins: Time-varying parameters, online/real-time, optimal filtering. When rolling wins: Simple, interpretable, robust to misspecification.",
    difficulty: 'advanced',
  },
  {
    id: 3,
    question:
      'Why is Extended Kalman Filter (EKF) needed for some finance problems while standard Kalman is not sufficient?',
    options: [
      'Financial data is always non-linear',
      'Standard Kalman assumes linear relationships; EKF handles non-linearity',
      'EKF is more computationally efficient',
      'Standard Kalman cannot handle missing data',
      'EKF required for high-frequency data',
    ],
    correctAnswer: 1,
    explanation:
      "EKF handles non-linear relationships. Standard Kalman: Linear state transition & observation equations. Many finance problems are NON-LINEAR: Option pricing (Black-Scholes), Volatility (GARCH in log-space), Stochastic volatility models. EKF: Linearizes non-linear functions using first-order Taylor approximation (Jacobian). Example: Estimate volatility σ where observation is return² → non-linear. Alternative: Unscented Kalman Filter (UKF) - better approximation for highly non-linear systems. Use standard Kalman when possible (faster, more stable). Use EKF/UKF when non-linearity significant. Finance applications: Dynamic Nelson-Siegel yield curve, stochastic vol models, option implied vol surface.",
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'In Kalman pairs trading, the innovation (prediction error) sequence should ideally be:',
    options: [
      'Auto-correlated with increasing variance',
      'White noise (uncorrelated, constant variance)',
      'Trending downward over time',
      'Perfectly zero at all times',
      'Follow GARCH process',
    ],
    correctAnswer: 1,
    explanation:
      "White noise (uncorrelated, constant variance). Innovation = Observation - Prediction = $y_t - \\hat{y}_t|_{t-1}$. If Kalman filter correctly specified: Innovations should be white noise. Why? Kalman extracts all predictable information → residuals contain only unpredictable noise. Diagnostics: Test innovations for: (1) Zero mean (unbiased), (2) Constant variance (homoskedastic), (3) No autocorrelation (Ljung-Box test). If innovations NOT white noise: Model misspecified! → Maybe need different state model, wrong noise covariances, missing variables. Application: Check innovation statistics to validate Kalman model before trading. If autocorrelated → leaving alpha on table OR model overfitting.",
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'Kalman smoothing vs filtering: What is the key difference?',
    options: [
      'Smoothing uses future information, filtering only uses past',
      'Smoothing is faster computationally',
      'Filtering provides better estimates',
      'They are the same algorithm',
      'Smoothing only works for linear systems',
    ],
    correctAnswer: 0,
    explanation:
      "Smoothing uses future information. Kalman filtering: Estimates state at time t using data up to time t → $\\hat{x}_t|_t$ (real-time, causal). Kalman smoothing: Estimates state at time t using ALL data (past AND future) → $\\hat{x}_t|_T$ where T > t (offline, non-causal). Smoothing is better (lower variance) but cannot be used for trading! Application: Use filtering for live trading (only past data available), Use smoothing for historical analysis/research. Algorithm: Forward pass (filter) + backward pass (smoother). Example: Estimate yesterday's beta with today's information → smoothing gives better estimate but cannot trade on it (look-ahead bias). Financial use: Smoothing for backtesting regime identification, filtering for actual trading signals.",
    difficulty: 'intermediate',
  },
];

