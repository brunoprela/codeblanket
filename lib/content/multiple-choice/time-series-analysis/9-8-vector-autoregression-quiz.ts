export const vectorAutoregressionMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'A VAR(2) model with 3 variables (X, Y, Z) has how many parameters per equation (excluding intercept)?',
    options: [
      '2 (the lag order)',
      '6 (2 lags × 3 variables)',
      '3 (one per variable)',
      '5 (2 lags + 3 variables)',
      '9 (3 variables × 3 equations)',
    ],
    correctAnswer: 1,
    explanation:
      "6 parameters per equation. VAR(p) with k variables: Each equation has k×p parameters. Here: k=3 variables, p=2 lags → 3×2 = 6 parameters per equation. Structure: $X_t = c + a_1 X_{t-1} + a_2 Y_{t-1} + a_3 Z_{t-1} + a_4 X_{t-2} + a_5 Y_{t-2} + a_6 Z_{t-2} + ε_t$. Total parameters in system: 3 equations × (6 + 1 intercept) = 21. This rapid parameter growth is why VAR needs sufficient data: Rule of thumb: Need T > k²p observations minimum. For VAR(2) with 3 variables: Need > 18 observations (preferably 10× = 180).",
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'Granger causality test shows X Granger-causes Y (p=0.01) but Y does NOT Granger-cause X (p=0.60). What can you conclude?',
    options: [
      'X causes Y in the true causal sense',
      'Past values of X help predict Y, but not vice versa',
      'Y causes X (reverse causality)',
      'X and Y are independent',
      'The test is invalid - must have bidirectional causality',
    ],
    correctAnswer: 1,
    explanation:
      "Past X predicts Y, but past Y doesn't predict X. Granger causality definition: X Granger-causes Y if past values of X provide statistically significant information about future Y, beyond what past Y alone provides. KEY: This is PREDICTIVE causality, NOT true causation! p=0.01 for X→Y: Reject H₀ (X doesn't help predict Y) → Past X improves Y forecast. p=0.60 for Y→X: Fail to reject H₀ → Past Y doesn't improve X forecast. Interpretation: Unidirectional Granger causality X→Y. Caveats: (1) Could be spurious (omitted variable Z causes both), (2) True causality might be Y→X but X is forward-looking, (3) Non-linear relationships not captured. Financial example: Interest rates Granger-cause stock prices (predictive) but true causation is complex (both react to Fed policy).",
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question:
      'In a VAR(1) with 2 variables, the companion matrix has eigenvalues 0.95 and 0.85. What does this indicate?',
    options: [
      'VAR is non-stationary (eigenvalue ≥ 1)',
      'VAR is stationary and stable (all eigenvalues < 1)',
      'VAR has unit root',
      'Need VAR(2) instead',
      'Cointegration is present',
    ],
    correctAnswer: 1,
    explanation:
      "VAR is stationary and stable. Stability condition: All eigenvalues of companion matrix must have modulus < 1. Here: max eigenvalue = 0.95 < 1 → Stable! Implications: (1) Shocks dissipate over time, (2) Variables don't explode, (3) Impulse responses converge to zero, (4) Forecasts converge to long-run means. Half-life of shocks: $t_{1/2} \\approx -\\log(2)/\\log(\\lambda_{max}) = -0.693/\\log(0.95) \\approx 13.5$ periods. If eigenvalue = 1: Unit root → non-stationary. If eigenvalue > 1: Explosive (rare in practice). Check stability: statsmodels VAR.fit().is_stable() or manually compute companion matrix eigenvalues. For VAR to be useful for forecasting: MUST be stable!",
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'An impulse response function (IRF) shows that a 1% shock to interest rates causes stock returns to decline by 0.5% initially, then by 0.3%, 0.1%, 0%, ... What does this pattern indicate?',
    options: [
      'Interest rates and stocks are cointegrated',
      'Negative impact that dissipates over time (monotonic decay)',
      'Interest rate shock has permanent negative effect',
      'Stock returns Granger-cause interest rates',
      'VAR model is misspecified',
    ],
    correctAnswer: 1,
    explanation:
      "Negative impact with monotonic decay. IRF interpretation: Period 0 (impact): -0.5% (immediate effect), Period 1: -0.3% (still negative but declining), Period 2: -0.1% (approaching zero), Period 3+: ≈0% (shock dissipates). This shows: (1) Negative relationship (higher rates → lower stocks), (2) Temporary effect (not permanent), (3) Stable VAR (shock dies out). Cumulative effect: -0.5% -0.3% -0.1% = -0.9% total. Compare to permanent effect: IRF would not converge to zero (unit root). Compare to overshooting: IRF might go positive before converging (oscillation). Financial interpretation: Rate hike negatively impacts stocks but market adjusts over ~3-5 periods. Use for trading: Initial move tradable, but effect temporary.",
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'Why might a VAR(1) with 50 variables be practically infeasible even with 10,000 observations?',
    options: [
      'Not enough data - need 50,000 observations',
      'Too many parameters (2,500+) causes overfitting and numerical issues',
      'VAR requires at least 2 lags',
      'Cannot estimate with more than 10 variables',
      'Computational cost is too high',
    ],
    correctAnswer: 1,
    explanation:
      "2,500+ parameters cause overfitting. VAR(1) with k=50 variables: Parameters per equation: 50 (lags) + 1 (intercept) = 51. Total parameters: 50 equations × 51 = 2,550 parameters. Issues: (1) Overfitting: Even with 10,000 observations, too many degrees of freedom → captures noise not signal, (2) Numerical instability: Matrix inversion difficult, (3) Out-of-sample performance poor, (4) Many spurious Granger causality relationships. Rule of thumb: k²p << T for reliable estimates. Here: k²p = 50² × 1 = 2,500 vs T = 10,000 → ratio too high. Solutions: (A) Bayesian VAR with shrinkage (Minnesota prior), (B) Factor VAR: Extract 5-10 factors, then VAR on factors, (C) Elastic net / lasso for variable selection, (D) Block VAR: Group related variables. Modern approach: BVAR or FAVAR for high-dimensional systems.",
    difficulty: 'advanced',
  },
];

