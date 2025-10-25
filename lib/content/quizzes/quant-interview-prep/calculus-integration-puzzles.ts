export const calculusIntegrationPuzzlesQuiz = [
  {
    id: 'cip-q-1',
    question:
      'Jane Street interview: "You have a function f (x) = x³ - 6x² + 9x + 1. Find all local maxima and minima, determine which is which, and sketch the function behavior. Then, if this represents profit as a function of quantity produced, what quantity maximizes profit and what is the maximum profit?" Provide complete first and second derivative analysis with economic interpretation.',
    sampleAnswer:
      "Complete optimization analysis: (1) Find critical points using first derivative: f (x) = x³ - 6x² + 9x + 1. f'(x) = 3x² - 12x + 9 = 3(x² - 4x + 3) = 3(x-1)(x-3). Critical points: f'(x) = 0 → x = 1 or x = 3. (2) Second derivative test: f''(x) = 6x - 12. At x=1: f''(1) = 6(1) - 12 = -6 < 0 → local maximum. At x=3: f''(3) = 6(3) - 12 = 6 > 0 → local minimum. (3) Evaluate function at critical points: f(1) = 1 - 6 + 9 + 1 = 5 (local max). f(3) = 27 - 54 + 27 + 1 = 1 (local min). (4) Sketch behavior: As x → -∞: f (x) → -∞ (negative cubic term dominates). As x → +∞: f (x) → +∞ (positive cubic term dominates). Function increases from -∞ to x=1 (local max at y=5), decreases from x=1 to x=3 (local min at y=1), then increases to +∞. (5) Economic interpretation: If f (x) represents profit vs quantity, the local maximum at x=1 gives profit of 5 (units/dollars). However, we must check boundary conditions. If quantity must be non-negative and there's a practical upper limit, the global maximum might be at a boundary. Assuming no constraints, x=1 maximizes profit at 5 units. But note: as x → ∞, profit → ∞, which is unrealistic (suggests model breaks down at high quantities - perhaps due to market saturation not captured). Practically, x=1 is the optimal operating quantity for realistic production levels. (6) Verification: test points: f(0) = 1, f(1) = 5 ✓, f(2) = 8-24+18+1 = 3, f(3) = 1 ✓, f(4) = 64-96+36+1 = 5. Sketch confirms: increase to (1,5), decrease to (3,1), increase thereafter.",
    keyPoints: [
      "Critical points: f'(x) = 3(x-1)(x-3) = 0 → x = 1, 3",
      "Second derivative: f''(1) = -6 < 0 (max), f''(3) = 6 > 0 (min)",
      'Local max at (1, 5), local min at (3, 1)',
      'Economic interpretation: x=1 maximizes profit at 5 units',
      'Check boundary conditions and model validity for large x',
    ],
  },
  {
    id: 'cip-q-2',
    question:
      'Citadel interview: "A stock price follows geometric Brownian motion: dS/dt = μS. You buy a call option with payoff max(S_T - K, 0). Express the expected payoff as an integral assuming S(t) = S₀e^(μt) and terminal time T. Then approximate the integral for small μT using Taylor expansion. Finally, relate this to the Black-Scholes formula intuition." Show complete integration steps and financial interpretation.',
    sampleAnswer:
      'Complete stochastic calculus problem: (1) Price evolution: dS/dt = μS → S(t) = S₀e^(μt) (exponential growth). At time T: S_T = S₀e^(μT). (2) Call payoff: C_T = max(S_T - K, 0). Expected payoff: E[C_T] = E[max(S₀e^(μT) - K, 0)]. For deterministic S_T (no randomness in this simplified model): if S₀e^(μT) > K, payoff = S₀e^(μT) - K; else 0. This is too simple - real model needs randomness. Let me add volatility: S_T ~ lognormal with mean S₀e^(μT). (3) With lognormal distribution: S_T = S₀exp(μT + σ√T·Z) where Z ~ N(0,1). E[C_T] = ∫_{-∞}^∞ max(S₀e^(μT+σ√T·z) - K, 0) · (1/√(2π))e^(-z²/2) dz. Payoff positive when S₀e^(μT+σ√T·z) > K, i.e., z > (ln(K/S₀) - μT)/(σ√T) = -d. E[C_T] = ∫_d^∞ [S₀e^(μT+σ√T·z) - K] · φ(z) dz, where φ(z) = (1/√(2π))e^(-z²/2). Split: E[C_T] = S₀e^(μT) ∫_d^∞ e^(σ√T·z) φ(z) dz - K ∫_d^∞ φ(z) dz. Second integral: ∫_d^∞ φ(z) dz = N(-d) where N is CDF of standard normal. First integral requires completing the square: ∫ e^(σ√T·z) e^(-z²/2) dz = e^((σ√T)²/2) ∫ e^(-(z-σ√T)²/2) dz = e^((σ√T)²/2) N(...). Full solution: E[C_T] = S₀e^(μT + σ²T/2)N(d₁) - K·N(d₂), where d₁, d₂ involve σ√T terms. (4) Small μT approximation: For μT << 1, e^(μT) ≈ 1 + μT. If at-the-money (S₀ ≈ K): E[C_T] ≈ (S₀ - K) + terms involving σ√T. The dominant term for ATM is ∝ σ√T, giving E[C_T] ≈ 0.4·S₀·σ√T (the mental math formula). (5) Black-Scholes connection: Full BS formula includes: risk-neutral drift (r instead of μ), discounting, exact distributional integrals. The core insight: option value ≈ intrinsic + time value, where time value ∝ σ√T. Interview takeaway: exponential growth, lognormal distribution, integration over payoff region, Taylor expansion for approximation.',
    keyPoints: [
      'Price dynamics: S(t) = S₀e^(μt) from dS/dt = μS',
      'Lognormal distribution: S_T = S₀exp(μT + σ√T·Z), Z~N(0,1)',
      'Expected payoff requires integration over Z > threshold',
      'Small μT approximation: e^(μT) ≈ 1 + μT',
      'ATM option value ≈ 0.4·S₀·σ√T from distribution integrals',
    ],
  },
  {
    id: 'cip-q-3',
    question:
      'Two Sigma interview: "You need to optimize portfolio allocation between two assets. Asset A has return r_A = 10% and variance σ_A² = 0.04. Asset B has return r_B = 8% and variance σ_B² = 0.02. Correlation ρ = 0.5. Allocate weight w to A, (1-w) to B. Find w that: (1) maximizes return, (2) minimizes variance, (3) maximizes Sharpe ratio (assume risk-free rate = 2%). Provide calculus-based solutions and optimal weights." Show complete derivative work and financial interpretation.',
    sampleAnswer:
      "Complete portfolio optimization: (1) Expected return: E[R] = w·r_A + (1-w)·r_B = w(0.10) + (1-w)(0.08) = 0.08 + 0.02w. To maximize: dE[R]/dw = 0.02 > 0, so return increases with w. Maximum return: w = 1 (all in asset A). Return = 10%. (2) Portfolio variance: σ_p² = w²σ_A² + (1-w)²σ_B² + 2w(1-w)ρσ_Aσ_B = w²(0.04) + (1-w)²(0.02) + 2w(1-w)(0.5)(0.2)(0.1414) = 0.04w² + 0.02(1-2w+w²) + 2w(1-w)(0.5)(0.0283) = 0.04w² + 0.02 - 0.04w + 0.02w² + 0.0283w - 0.0283w² = (0.04 + 0.02 - 0.0283)w² + (-0.04 + 0.0283)w + 0.02 = 0.0317w² - 0.0117w + 0.02. To minimize variance: dσ_p²/dw = 2(0.0317)w - 0.0117 = 0 → 0.0634w = 0.0117 → w = 0.185. At w=0.185: σ_p² ≈ 0.0317(0.034) - 0.0117(0.185) + 0.02 ≈ 0.019. Minimum variance portfolio: w ≈ 18.5% in A, 81.5% in B. (3) Sharpe ratio: SR = (E[R] - r_f) / σ_p = (0.08 + 0.02w - 0.02) / √(0.0317w² - 0.0117w + 0.02) = (0.06 + 0.02w) / √(0.0317w² - 0.0117w + 0.02). To maximize: d(SR)/dw = 0. Using quotient rule: d(SR)/dw = [0.02·σ_p - (0.06+0.02w)·(dσ_p/dw)] / σ_p². Set numerator = 0: 0.02·σ_p = (0.06+0.02w)·(0.0317w - 0.00585)/σ_p. This is complex; numerical solution or simplification needed. Alternatively, maximum Sharpe is typically between min-variance and max-return. Numerically (or via optimization): optimal w ≈ 0.6-0.7 for maximum Sharpe. Let me recalculate more carefully with correct σ values: σ_A = 0.2, σ_B = 0.1414. Using exact formula for minimum variance with correlation: w_min_var = (σ_B² - ρσ_Aσ_B) / (σ_A² + σ_B² - 2ρσ_Aσ_B) = (0.02 - 0.5·0.0283) / (0.04 + 0.02 - 2·0.5·0.0283) = (0.02 - 0.01415) / (0.06 - 0.0283) = 0.00585 / 0.0317 ≈ 0.185 ✓. For Sharpe ratio, optimal w depends on (r_A-r_f), (r_B-r_f), variances, correlation. General formula exists but complex. Interview approach: state the 3 problems clearly, solve min-variance exactly, note max-return is corner solution, numerically estimate max-Sharpe or state it's between the two extremes.",
    keyPoints: [
      'Max return: w=1 (all in asset A), return=10%',
      'Min variance: w≈0.185 using formula w=(σ_B²-ρσ_Aσ_B)/(σ_A²+σ_B²-2ρσ_Aσ_B)',
      'Sharpe ratio: requires solving quotient rule derivative = 0',
      'Optimal Sharpe typically between min-var and max-return',
      'Financial tradeoff: higher return (A) vs lower risk (B)',
    ],
  },
];
