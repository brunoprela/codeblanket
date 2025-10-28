export default {
  id: 'fin-m15-s2-quiz',
  title: 'Value at Risk (VaR) Methods - Quiz',
  questions: [
    {
      id: 1,
      question:
        'Historical simulation VaR uses a 3-year lookback period (756 trading days). What is the main limitation of this approach during a market regime change?',
      options: [
        'Computational complexity makes it too slow',
        'Historical data may not reflect current market volatility and correlations',
        'It requires assumptions about return distributions',
        'It cannot handle derivatives with non-linear payoffs',
      ],
      correctAnswer: 1,
      explanation:
        "The main limitation of historical simulation is that it relies on the past 3 years of data to predict tomorrow's risk. If market conditions have changed (e.g., transitioning from low to high volatility), historical data will underestimate current risk. For example, using 2005-2007 data (calm period) to calculate VaR in early 2008 would severely underestimate risk as the crisis began. Option A is wrong—historical simulation is actually computationally fast (just sorting historical returns). Option C is wrong—historical simulation makes NO distributional assumptions (that's parametric VaR). Option D is wrong—historical simulation handles non-linear instruments fine by revaluing them at historical scenarios. The key issue is stale data not reflecting current regime.",
    },
    {
      id: 2,
      question:
        'A portfolio has 99% parametric VaR of $50M assuming normal returns. Historical VaR (same confidence) is $75M. What does this suggest?',
      options: [
        'There is a calculation error—both methods should give the same answer',
        'Returns have fat tails (more extreme events than normal distribution predicts)',
        'The portfolio has increased in size',
        'Historical simulation uses a longer time period',
      ],
      correctAnswer: 1,
      explanation:
        "When historical VaR significantly exceeds parametric VaR (50% higher: $75M vs $50M), this indicates fat tails—the actual historical data shows more extreme losses than a normal distribution would predict. Parametric VaR assumes normal distribution where 99th percentile = 2.33 standard deviations. But if actual returns have fat tails (common in financial markets), historical VaR captures the true tail better. Option A is wrong—different methods give different answers (that's why we use multiple methods). Option C is irrelevant—both methods use current portfolio, so size is the same. Option D is wrong—the time period affects both methods but doesn't explain systematic difference. This divergence is a warning sign: normality assumption is violated, parametric VaR is too optimistic.",
    },
    {
      id: 3,
      question:
        'Monte Carlo VaR simulation requires 100,000 scenarios and takes 60 seconds to compute. For real-time pre-trade checks (<100ms), what is the best approach?',
      options: [
        'Use faster computers to reduce Monte Carlo time to <100ms',
        'Use parametric VaR for pre-trade checks; Monte Carlo for EOD official VaR',
        'Reduce scenarios to 100 for speed',
        'Eliminate pre-trade checks as they slow trading',
      ],
      correctAnswer: 1,
      explanation:
        "The solution is layering multiple VaR methods: fast approximations (parametric or incremental VaR) for real-time pre-trade checks, and accurate but slow methods (Monte Carlo) for end-of-day official VaR. Option A (faster computers) won't help enough—even with 100x faster computers, 60 seconds becomes 0.6 seconds, still too slow. Option C (reduce scenarios to 100) destroys accuracy—100 scenarios can't reliably estimate 99th percentile (need 1 in 100 extreme cases). Option D is unacceptable—pre-trade checks prevent limit breaches. The trade-off is deliberate: accept 95% accuracy for real-time (<100ms parametric), get 99%+ accuracy for official reporting (60s Monte Carlo). Compare hourly to prevent drift.",
    },
    {
      id: 4,
      question:
        'A firm backtests its 99% daily VaR over 250 days. VaR is exceeded 8 times. What should the firm conclude?',
      options: [
        'The VaR model is accurate (8 is close to expected 2.5 breaches)',
        'The VaR model significantly underestimates risk and should be recalibrated',
        'This is within normal statistical variation—no action needed',
        'The firm should switch to 95% VaR for fewer breaches',
      ],
      correctAnswer: 1,
      explanation:
        "With 99% VaR, we expect breaches 1% of the time, so over 250 days we expect 2.5 breaches. Observing 8 breaches (3.2%) is 3.2x the expected rate—this is statistically significant and indicates the model underestimates risk. Regulatory tests (Basel traffic light approach) would flag this as a problem. Option A is wrong—8 is not close to 2.5. Option C is wrong—while some variation is expected (binomial distribution), 8 breaches has p-value <0.01, rejecting the null hypothesis that VaR is accurate. Option D misses the point—switching to 95% VaR doesn't fix the underlying model problem, it just changes the threshold. The firm must investigate why VaR underestimated: wrong volatility estimate? Fat tails not captured? Regime change? Then recalibrate.",
    },
    {
      id: 5,
      question:
        'Which statement about the square-root-of-time rule for scaling VaR is correct?',
      options: [
        'It is always accurate for any time horizon',
        'It assumes IID returns and fails when returns are autocorrelated or volatility clusters',
        'It should only be used to scale from daily to weekly VaR',
        'It overstates long-horizon VaR due to mean reversion',
      ],
      correctAnswer: 1,
      explanation:
        "The square-root-of-time rule (10-day VaR = 1-day VaR × √10) assumes returns are independent and identically distributed (IID). This breaks down when: (1) Returns are autocorrelated (momentum/reversal effects), (2) Volatility clusters (high vol days follow high vol days), (3) Fat tails exist. Option A is wrong—it's an approximation that fails under these conditions. Option C is wrong—it can be used for any horizon, but accuracy degrades for longer periods. Option D has it backwards—the rule typically UNDERSTATES long-horizon VaR because it ignores volatility clustering (periods of sustained high volatility). In practice, for horizons >10 days, firms often use full simulation rather than scaling. For example, 10-day VaR is often >√10 times 1-day VaR due to persistence in volatility.",
    },
  ],
} as const;
