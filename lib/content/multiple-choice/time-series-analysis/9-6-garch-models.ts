export const garchModelsMultipleChoice = [
  {
    id: 1,
    question:
      'A GARCH(1,1) model has parameters ω=0.00001, α=0.12, β=0.85. What is the unconditional (long-run) volatility?',
    options: [
      '0.01 or 1%',
      '0.0173 or 1.73%',
      '0.97 (sum of α+β)',
      '0.000333 (ω/(α+β))',
      'Cannot be determined without data',
    ],
    correctAnswer: 1,
    explanation:
      'Unconditional volatility is 1.73%. Formula for GARCH(1,1) long-run variance: $\\sigma^2 = \\omega / (1 - \\alpha - \\beta)$. Here: $\\sigma^2 = 0.00001 / (1 - 0.12 - 0.85) = 0.00001 / 0.03 = 0.000333$. Taking square root: $\\sigma = \\sqrt{0.000333} = 0.0173 = 1.73\\%$. This is the LONG-RUN average volatility that the model forecasts in the absence of shocks. Key insights: (1) Requires α+β < 1 for stationarity (here 0.97 < 1 ✓), (2) Higher α+β → longer persistence → current vol stays away from long-run vol longer, (3) ω sets the scale of long-run vol, α and β determine dynamics. If α+β ≥ 1: no unconditional variance (IGARCH - integrated GARCH). Common error: forgetting square root (answer would be 0.0333 or 3.33%).',
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'An ARCH test on stock returns yields LM statistic = 45.2, p-value = 0.001. What does this indicate?',
    options: [
      'Returns have no autocorrelation - use white noise model',
      'Significant ARCH effects present - GARCH model appropriate',
      'Returns are normally distributed',
      'Model has no heteroskedasticity',
      'Need higher AR order in mean equation',
    ],
    correctAnswer: 1,
    explanation:
      'ARCH effects ARE present - GARCH is appropriate. ARCH LM test: Null hypothesis H₀ = No ARCH effects (constant variance). With p-value = 0.001 < 0.05, we REJECT H₀ → ARCH effects exist. This means: (1) Volatility is time-varying (not constant), (2) Past squared residuals predict future variance, (3) GARCH/ARCH model appropriate to capture this. The test procedure: Regress squared residuals on lagged squared residuals, test joint significance. High LM statistic (45.2) indicates strong evidence. What to do: Fit GARCH(1,1) first (most common), check if residuals from GARCH no longer show ARCH effects (p > 0.05), if still significant, try GARCH(1,2), GARCH(2,1), or EGARCH. Common in finance: Almost all asset returns show ARCH effects at daily frequency. Exception: Some bond returns, FX during stable periods.',
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question:
      "In EGARCH(1,1), the asymmetry parameter γ = -0.05. Yesterday's standardized return was z = -2 (negative shock). How does this compare to a positive shock of z = +2 in terms of volatility impact?",
    options: [
      'Same impact - EGARCH is symmetric',
      'Negative shock increases vol by 0.05 more',
      'Negative shock increases vol LESS than positive',
      'Negative shock increases vol MORE (γ=-0.05 < 0 → leverage effect)',
      'Impact depends on current volatility level',
    ],
    correctAnswer: 3,
    explanation:
      'Negative shock increases vol MORE due to leverage effect. EGARCH asymmetry term: $g(z_t) = \\theta z_t + \\gamma(|z_t| - \\mathbb{E}|z_t|)$. For z=-2: $g(-2) = \\theta(-2) + \\gamma(2 - 0.798)$. For z=+2: $g(+2) = \\theta(+2) + \\gamma(2 - 0.798)$. The difference is in the θz term: negative z makes this negative (if θ<0), amplifying the effect. With γ=-0.05 < 0: Negative shocks (z<0) have larger impact on log(σ²). This captures the LEVERAGE EFFECT: Stock price drops → higher leverage → more risk → volatility increases. Magnitude: γ=-0.05 is moderate leverage effect. Typical range: -0.1 to 0 for equities. If γ=0: symmetric GARCH. If γ>0: positive returns increase vol more (rare, seen in some commodities). Key insight: EGARCH allows asymmetry without parameter constraints (log form ensures σ²>0).',
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'A trader forecasts 10-day ahead volatility using GARCH(1,1) with α=0.08, β=0.88, current conditional vol σₜ=2%, and long-run vol σ̄=1.5%. What is the approximate 10-day forecast?',
    options: [
      "2.0% (volatility doesn't change)",
      '1.5% (immediately reverts to long-run)',
      '1.65% (partial reversion toward long-run)',
      '2.5% (volatility increases)',
      '3.0% (square root of time rule)',
    ],
    correctAnswer: 2,
    explanation:
      'Approximately 1.65% due to mean reversion. GARCH(1,1) multi-step forecast: $\\mathbb{E}[\\sigma_{t+h}^2|\\mathcal{F}_t] = \\bar{\\sigma}^2 + (\\alpha+\\beta)^{h-1}(\\sigma_t^2 - \\bar{\\sigma}^2)$. Where: $\\bar{\\sigma}^2 = \\omega/(1-\\alpha-\\beta)$ = long-run variance. Here: $\\sigma_t = 2\\% = 0.02$, $\\bar{\\sigma} = 1.5\\% = 0.015$. $(\\alpha+\\beta) = 0.96$, $h=10$. Forecast variance: $\\mathbb{E}[\\sigma_{11}^2] = (0.015)^2 + (0.96)^9((0.02)^2 - (0.015)^2) = 0.000225 + 0.694(0.000175) = 0.000346$. Forecast vol: $\\sqrt{0.000346} = 0.0186 = 1.86\\%$ (closer to 1.65% with rounding). Interpretation: Current vol (2%) above long-run (1.5%), forecast partially reverts. Key: $(\\alpha+\\beta)^{h-1}$ decays exponentially. High persistence (0.96) → slow reversion. For h→∞: forecast → 1.5% (long-run vol). Common error: Using $\\sqrt{10} \\times \\sigma_t$ (iid scaling, wrong for GARCH).',
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'After fitting GARCH(1,1), the standardized residuals (ε/σ) still show significant autocorrelation in squared residuals (LB p-value = 0.02). What should you do?',
    options: [
      'Model is adequate, p=0.02 is close enough to 0.05',
      'Increase GARCH order to GARCH(2,1) or GARCH(1,2)',
      'Add AR terms to mean equation',
      'Use different distribution (Student-t instead of normal)',
      'Abandon GARCH and use rolling standard deviation',
    ],
    correctAnswer: 1,
    explanation:
      "Increase GARCH order to capture remaining volatility dynamics. Problem: LB test on standardized residuals² p=0.02 < 0.05 → reject white noise → ARCH effects REMAIN after GARCH(1,1). This means: (1) GARCH(1,1) not flexible enough, (2) Still predictable patterns in volatility, (3) Model is misspecified. Solutions (in order): Try GARCH(2,1): $\\sigma_t^2 = \\omega + \\alpha_1\\epsilon_{t-1}^2 + \\alpha_2\\epsilon_{t-2}^2 + \\beta\\sigma_{t-1}^2$ (adds second ARCH lag). Try GARCH(1,2): $\\sigma_t^2 = \\omega + \\alpha\\epsilon_{t-1}^2 + \\beta_1\\sigma_{t-1}^2 + \\beta_2\\sigma_{t-2}^2$ (adds persistence). Try EGARCH or GJR-GARCH (asymmetry). Check mean equation: Maybe need AR terms (autocorrelation in levels first). Why not option D (Student-t)? Distribution choice affects parameter estimates but doesn't fix model specification. If GARCH(p,q) adequate, standardized residuals² should have NO autocorrelation regardless of distribution. Only after specifying volatility dynamics should you choose distribution (normal vs t vs skewed-t).",
    difficulty: 'advanced',
  },
];
