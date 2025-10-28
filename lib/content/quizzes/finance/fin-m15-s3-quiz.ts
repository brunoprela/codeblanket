export default {
  id: 'fin-m15-s3-quiz',
  title: 'Conditional Value at Risk (CVaR) - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A portfolio has 99% VaR of $100M and 99% CVaR of $150M. What does this tell you?',
      options: [
        'There is a calculation error—CVaR cannot exceed VaR',
        'The tail is severe—average loss in worst 1% is 50% worse than threshold',
        'CVaR should be used instead of VaR for all purposes',
        'The portfolio should be reduced by $50M',
      ],
      correctAnswer: 1,
      explanation:
        "CVaR (Expected Shortfall) is always ≥ VaR because it measures the average loss in the tail, not just the threshold. CVaR of $150M vs VaR of $100M means: when losses exceed the 99th percentile, they average $150M—50% worse than the threshold. This indicates a severe tail (fat tail distribution). If returns were normally distributed, CVaR would only be ~15-20% higher than VaR. The 50% gap reveals extreme tail events are common. Option A is wrong—CVaR > VaR is expected. Option C is too absolute—both metrics have uses (VaR for regulatory, CVaR for risk management). Option D doesn't follow—CVaR informs decisions but doesn't directly dictate size. This gap should trigger: stress testing, tail hedging consideration, understanding concentration risks.",
    },
    {
      id: 2,
      question:
        'Why is CVaR considered a "coherent" risk measure while VaR is not?',
      options: [
        'CVaR is easier to calculate than VaR',
        'CVaR satisfies sub-additivity (diversification always reduces risk); VaR does not',
        'CVaR uses a higher confidence level than VaR',
        'CVaR is required by regulators while VaR is optional',
      ],
      correctAnswer: 1,
      explanation:
        'CVaR is coherent because it satisfies four axioms, including sub-additivity: CVaR(Portfolio A + B) ≤ CVaR(A) + CVaR(B). This ensures diversification always reduces risk. VaR can violate sub-additivity—combining two portfolios can increase VaR due to extreme tail events. Option A is wrong—CVaR is actually harder to calculate (must compute expected value in tail). Option C is wrong—both can use any confidence level. Option D is backwards—VaR is required by Basel; CVaR is optional (though recommended). Sub-additivity matters because risk managers want to encourage diversification. A non-sub-additive measure can show that diversifying INCREASES risk, which is perverse. This is why academics prefer CVaR despite industry using VaR (regulatory inertia).',
    },
    {
      id: 3,
      question:
        'A portfolio optimization uses CVaR as the objective function. Compared to minimizing VaR, what result would you expect?',
      options: [
        'Identical portfolios because VaR and CVaR measure the same thing',
        'Lower tail risk (fewer extreme losses) at the cost of slightly higher VaR',
        'Higher expected return for same VaR',
        'Concentration in fewer assets',
      ],
      correctAnswer: 1,
      explanation:
        'Optimizing CVaR (minimize expected tail loss) produces portfolios with better tail behavior than optimizing VaR. VaR optimization only cares about the threshold—it can produce portfolios with acceptable VaR but catastrophic losses beyond. CVaR optimization avoids extreme tail events, resulting in lower CVaR but potentially slightly higher VaR (the threshold might be worse, but beyond it is much better). Option A is wrong—they produce different portfolios. Option C is unclear—return depends on constraints. Option D is backwards—CVaR optimization tends to diversify more because CVaR penalizes concentration in tail risk. Example: VaR optimizer might accept exposure to tail event with 0.5% probability (below 1% threshold). CVaR optimizer avoids it because it increases expected tail loss.',
    },
    {
      id: 4,
      question:
        'For a portfolio with normal returns, 99% CVaR is approximately what multiple of 99% VaR?',
      options: [
        '0.5x (CVaR is half of VaR)',
        '1.0x (CVaR equals VaR)',
        '1.2x (CVaR is 20% higher)',
        '2.0x (CVaR is double VaR)',
      ],
      correctAnswer: 2,
      explanation:
        "For normally distributed returns, 99% CVaR ≈ 1.2× VaR (20% higher). This is because VaR is the 99th percentile (2.33σ), while CVaR is the expected value beyond that, which for normal distribution is approximately 2.67σ. The ratio 2.67/2.33 ≈ 1.15-1.20. If your portfolio shows CVaR >> 1.2× VaR (e.g., 1.5× or higher), this indicates fat tails—tail events are worse than normal distribution predicts. Option A is impossible (CVaR can't be less than VaR). Option B is only true if all tail losses exactly equal VaR (unrealistic). Option D (2.0×) would indicate extreme fat tails. This ratio is a useful diagnostic: ratio ≈ 1.2 suggests normality; ratio > 1.5 suggests fat tails; investigate further.",
    },
    {
      id: 5,
      question:
        'A risk manager reports "99% CVaR is $200M" to the board. The board asks: "What does this mean in simple terms?" What is the best answer?',
      options: [
        '"There is a 1% chance we will lose more than $200M"',
        '"On our worst days (worst 1%), we lose $200M on average"',
        '"The maximum possible loss is $200M"',
        '"We need $200M of capital"',
      ],
      correctAnswer: 1,
      explanation:
        'The clearest explanation of CVaR: "On our worst days (worst 1% of days), we lose $200M on average." This conveys the conditional nature (given we\'re in the tail) and the expectation (average, not maximum). Option A confuses CVaR with VaR—CVaR is not a threshold but an average beyond threshold. Option C is wrong—CVaR is NOT maximum loss (which could be much higher). Option D is related but imprecise—capital should cover unexpected losses, which relates to CVaR but isn\'t identical. Good risk communication to boards requires plain language. Compare: VaR = "1 day in 100, losses exceed X"; CVaR = "On those worst days, losses average Y". CVaR gives more complete picture of tail risk, which is why sophisticated boards should ask for both metrics.',
    },
  ],
} as const;
