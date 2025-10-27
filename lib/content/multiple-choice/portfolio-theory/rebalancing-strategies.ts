export const rebalancingStrategiesMC = {
  id: 'rebalancing-strategies-mc',
  title: 'Rebalancing Strategies - Multiple Choice',
  questions: [
    {
      id: 'rs-mc-1',
      type: 'multiple-choice' as const,
      question: 'The "rebalancing bonus" primarily comes from:',
      options: [
        'Transaction cost savings',
        'Tax benefits',
        'Buying low and selling high through mean reversion',
        'Leverage effects',
      ],
      correctAnswer: 2,
      explanation:
        "Answer: C. Rebalancing bonus arises from systematically buying assets after they fall (cheap) and selling after they rise (expensive). Captures mean reversion and volatility. Formula: ≈ 0.5×σ²×(1-ρ). Typical magnitude: 0.3-0.7% annually. NOT from transaction costs (A - that's a cost, not benefit), not tax benefits (B - rebalancing often increases taxes), not leverage (D - no leverage required).",
    },
    {
      id: 'rs-mc-2',
      type: 'multiple-choice' as const,
      question:
        'Portfolio drifted from target 60/40 to actual 68/32 due to equity rally. Rebalancing threshold is 5%. Should you rebalance?',
      options: [
        'Yes, equity drift is 8% which exceeds 5% threshold',
        "No, equity drift is 8% but it's measured wrong",
        'Yes, total drift is 16% which exceeds threshold',
        'No, use absolute deviation: |68-60| = 8%, but check if portfolio volatility changed significantly',
      ],
      correctAnswer: 3,
      explanation:
        'Answer: D. Equity drift = |68-60| = 8 percentage points. Threshold is typically 5pp, so 8pp > 5pp suggests rebalancing. However, best practice: also check if portfolio risk changed significantly. If portfolio volatility went from 12% to 13.5% (within tolerance), might skip. If it went to 15%, definitely rebalance. The 5% threshold is a guideline, not absolute rule. Context matters: costs, taxes, market conditions.',
    },
    {
      id: 'rs-mc-3',
      type: 'multiple-choice' as const,
      question:
        'Quarterly calendar rebalancing typically generates how much turnover annually?',
      options: ['10-20%', '30-50%', '60-80%', '100-150%'],
      correctAnswer: 1,
      explanation:
        "Answer: B. Quarterly rebalancing typically generates 30-50% annual turnover. Each rebalance might involve trading 10-15% of portfolio (selling 5-7% of outperformers, buying 5-7% of underperformers), × 4 quarters = 40-60% annually. Not 10-20% (A - too low, that's annual rebalancing), not 60-80% (C - that's monthly), not 100-150% (D - that's high-frequency strategies). Actual turnover depends on volatility and correlations.",
    },
    {
      id: 'rs-mc-4',
      type: 'multiple-choice' as const,
      question: 'Tax-efficient rebalancing prioritizes:',
      options: [
        'Selling winners in taxable accounts to rebalance',
        'Rebalancing first in tax-advantaged accounts (IRA/401k)',
        'Frequent rebalancing to capture small drifts',
        'Using derivatives to avoid capital gains',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. Rebalance in IRA/401k first (no capital gains tax). Then use new contributions (tax-free). Then tax-loss harvest in taxable. Last resort: sell winners (triggers tax). NOT selling winners first (A - that's tax-inefficient!), not frequent rebalancing (C - increases taxable events), derivatives (D - possible but complex, not primary strategy). Hierarchy: IRA first, contributions second, TLH third, sales last.",
    },
    {
      id: 'rs-mc-5',
      type: 'multiple-choice' as const,
      question:
        'Threshold rebalancing with 5% bands is generally superior to monthly calendar rebalancing because:',
      options: [
        'It generates higher returns',
        'It has lower tracking error',
        'It captures similar benefits with ~60% lower transaction costs',
        'It eliminates the need for monitoring',
      ],
      correctAnswer: 2,
      explanation:
        "Answer: C. 5% threshold triggers ~15-20 rebalances over 10 years vs ~120 for monthly, reducing costs 60-85% while capturing 85-90% of rebalancing benefit. Doesn't generate higher GROSS returns (A - similar), doesn't have lower TE (B - actually slightly higher), doesn't eliminate monitoring (D - must check regularly for threshold breach). The advantage is cost efficiency: similar benefit, much lower cost.",
    },
  ],
};
