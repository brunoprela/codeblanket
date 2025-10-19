/**
 * Quiz questions for Stochastic Processes for Finance section
 */

export const stochasticprocessesfinanceQuiz = [
  {
    id: 'q1',
    question:
      'Explain why Geometric Brownian Motion is preferred over simple Random Walk for modeling stock prices. What are the key differences?',
    hint: 'Think about negative prices and percentage changes.',
    sampleAnswer:
      'Random Walk: S_t = S_0 + Σ X_i can go negative, which is impossible for stock prices. Also models absolute changes (±$1), but stocks move in percentage terms. GBM: dS = μS dt + σS dW models percentage changes (the S multipliers make returns proportional to price) and ensures S_t > 0 always (exponential form). Key differences: (1) RW: additive noise, can be negative. GBM: multiplicative noise, always positive. (2) RW: returns not stationary (variance of price level grows indefinitely). GBM: log-returns stationary (log S_t ~ Normal). (3) RW: volatility constant in absolute terms. GBM: volatility proportional to price. (4) GBM matches empirical properties: stock returns approximately log-normal, percentage changes more relevant than absolute. Example: $10 stock moving ±$1 (10%) is different from $100 stock moving ±$1 (1%). GBM captures this correctly. This is why Black-Scholes and most option pricing models use GBM.',
    keyPoints: [
      'Random Walk can go negative, GBM cannot',
      'GBM models percentage changes, RW models absolute',
      'GBM returns are log-normal, realistic for stocks',
      'GBM ensures price always positive via exponential form',
      'GBM foundation for Black-Scholes and options pricing',
    ],
  },
  {
    id: 'q2',
    question:
      'Mean-reverting processes like Ornstein-Uhlenbeck are used for pairs trading. Explain the intuition and how mean reversion parameter θ affects trading strategy.',
    sampleAnswer:
      "Mean reversion: dX = θ(μ-X)dt + σdW. When X > μ (above mean), drift term -θ(X-μ) is negative, pulling X down toward μ. When X < μ, drift is positive, pulling up. Pairs trading application: Model spread S_t = Price_A - β·Price_B as mean-reverting. When spread > μ (stocks diverged), expect convergence: short stock A, long stock B. When spread < μ, reverse. Parameter θ effects: (1) Large θ: fast mean reversion, spread returns quickly. Trade more frequently with shorter holding periods. (2) Small θ: slow reversion, longer holding periods needed. (3) If θ ≈ 0, not mean-reverting - pairs trade won't work! Testing strategy: (1) Verify spread is stationary (ADF test). (2) Estimate θ, μ, σ from historical data. (3) Set thresholds: enter when |X-μ| > k·σ. (4) Exit when X returns to μ. Risk: If mean reversion breaks down (regime change), losses accumulate. Always use stop-losses.",
    keyPoints: [
      'Mean reversion: process tends toward long-term mean μ',
      'θ controls speed: larger θ = faster reversion',
      'Pairs trading: exploit spread mean reversion',
      'Buy when below mean, sell when above',
      'Must test for stationarity before trading',
    ],
  },
  {
    id: 'q3',
    question:
      'Poisson processes model discrete events like defaults or jumps. How does this differ from continuous processes like GBM, and why might you combine them?',
    sampleAnswer:
      'Poisson process: Discrete events at random times with constant rate λ. Events are jumps - instantaneous changes. GBM: Continuous diffusion - smooth random walk with drift. No jumps, always continuous path. Key differences: (1) Poisson: discontinuous, discrete events. GBM: continuous, differentiable. (2) Poisson: rare events (defaults, crashes). GBM: everyday volatility. (3) Poisson: heavy tails. GBM: normal returns. Why combine (Jump Diffusion Model): Real markets have both continuous volatility AND occasional jumps (crashes, earnings surprises, news). Model: dS = μS dt + σS dW + S dJ where dJ is Poisson jump process. Captures: (1) Normal times: continuous diffusion (GBM). (2) Event times: sudden jumps (Poisson). (3) Better fits empirical data: heavy tails, volatility clustering. Applications: (1) Credit derivatives: model defaults (Poisson). (2) Options pricing: accounts for jump risk. (3) Risk management: models tail events. Combining gives more realistic price processes than pure GBM.',
    keyPoints: [
      'Poisson: discrete jump events at random times',
      'GBM: continuous diffusion process',
      'Real markets have both: continuous + jumps',
      'Jump-diffusion combines both processes',
      'Better captures crashes and tail events',
    ],
  },
];
