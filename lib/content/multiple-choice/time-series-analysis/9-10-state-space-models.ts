export const stateSpaceModelsMultipleChoice = [
  {
    id: 1,
    question:
      'In a local level model, increasing the state noise variance (η) relative to observation noise (ε) causes the filtered estimate to:',
    options: [
      'Become smoother (less responsive)',
      'Become more responsive to observations',
      'Remain unchanged',
      'Diverge from observations',
      'Become more accurate',
    ],
    correctAnswer: 1,
    explanation:
      "Become more responsive. Signal-to-noise ratio: SNR = Var(η)/Var(ε). Higher SNR → Kalman filter believes observations contain useful state changes → follows data more closely. Lower SNR → believes observations are mostly noise → smooths heavily. Trade-off: High η → responsive but noisy, Low η → smooth but lagged. Optimal: Estimate variances from data using maximum likelihood.",
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'State space representation allows ARIMA(p,d,q) to be written as:',
    options: [
      'ARIMA cannot be written in state space form',
      'State equation only (no observation equation needed)',
      'First-order VAR with augmented state vector',
      'Second-order differential equation',
      'Non-linear dynamic system',
    ],
    correctAnswer: 2,
    explanation:
      "First-order VAR with augmented state vector. Any ARIMA(p,d,q) can be written as: State: X_t = [y_t, y_{t-1}, ..., ε_{t-1}, ...]^T, Transition: X_t = F·X_{t-1} + G·ε_t (first-order Markov), Observation: y_t = H·X_t. This representation enables: Kalman filtering, missing data handling, multivariate extensions. All linear time series models have state space form!",
    difficulty: 'advanced',
  },
  {
    id: 3,
    question:
      'What is the main advantage of state space over reduced-form models like ARIMA?',
    options: [
      'State space is always more accurate',
      'Structural interpretation and unobserved components',
      'State space has fewer parameters',
      'Computational efficiency',
      'Better short-term forecasts',
    ],
    correctAnswer: 1,
    explanation:
      "Structural interpretation and unobserved components. State space explicitly models: latent states (trend, cycle, volatility), economic structure (Phillips curve, term structure), time-varying parameters. ARIMA: reduced-form, forecasting-focused, no structural interpretation. Use state space when: need decomposition, theory-driven models, latent variables. Use ARIMA for: pure forecasting, parsimony, simple implementation.",
    difficulty: 'intermediate',
  },
  {
    id: 4,
    question:
      'In financial applications, why use state space for yield curve modeling?',
    options: [
      'Yield curves are always non-stationary',
      'Latent factors (level, slope, curvature) drive yields',
      'State space is required for arbitrage-free models',
      'ARIMA cannot model multiple maturities',
      'Computational speed',
    ],
    correctAnswer: 1,
    explanation:
      "Latent factors drive yields. Nelson-Siegel-Svensson model: Yield(τ) = β₁ + β₂·f(τ) + β₃·g(τ). State: [β₁_t, β₂_t, β₃_t] = [Level, Slope, Curvature]. Observation: [Y₁_t, Y₂_t, ..., Y_N_t] = observed yields. State space: Estimates time-varying factors, handles missing maturities, enforces no-arbitrage. Applications: Trading relative value, risk management, monetary policy analysis.",
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'Maximum likelihood estimation in state space models optimizes:',
    options: [
      'Sum of squared residuals',
      'Kalman filter innovation likelihood',
      'State transition probabilities',
      'Observation matrix elements',
      'Prior distributions',
    ],
    correctAnswer: 1,
    explanation:
      "Kalman filter innovation likelihood. Innovations: ν_t = y_t - E[y_t|past]. Under normality: ν_t ~ N(0, S_t). Log-likelihood: ℓ = Σ log p(ν_t|S_t). Optimization: Find parameters (F, G, H, Q, R) that maximize ℓ. Kalman filter: Recursively computes innovations and their variances → efficient likelihood evaluation. Numerical optimization (BFGS, Newton) finds MLE. Standard errors via Hessian or bootstrap.",
    difficulty: 'advanced',
  },
];

