export const backtestingPortfoliosMC = {
  id: 'backtesting-portfolios-mc',
  title: 'Backtesting Portfolios - Multiple Choice',
  questions: [
    {
      id: 'bp-mc-1',
      type: 'multiple-choice' as const,
      question:
        'Strategy with gross return 18%, turnover 200%, transaction costs 12 bps per trade. Net return?',
      options: ['17.76%', '15.60%', '16.80%', '14.40%'],
      correctAnswer: 1,
      explanation:
        'Cost = Turnover × Cost per trade = 200% × 0.12% = 0.24% × 200 = 2.4%. Wait, let me recalculate: 200% turnover means trading 2× portfolio value annually. Cost = 200% × 0.0012 = 0.024 = 2.4%. Net return = 18% - 2.4% = 15.6%. Answer: B. High turnover (200%) destroys 2.4% annually in costs, a 13% reduction in returns! This illustrates how costs cripple high-turnover strategies.',
    },
    {
      id: 'bp-mc-2',
      type: 'multiple-choice' as const,
      question:
        'Backtest optimization uses 10 parameters on 120 monthly observations. What is the degrees of freedom concern?',
      options: [
        'No concern, 120 observations is plenty',
        'Marginal concern, 12 observations per parameter is acceptable',
        'Serious concern, need 30+ observations per parameter',
        'Invalid backtest, need T > N²',
      ],
      correctAnswer: 2,
      explanation:
        'Answer: C. Rule of thumb: need 30+ observations per parameter for reliable estimation. 120/10 = 12 observations per parameter is insufficient. High overfitting risk - model will fit noise. Should reduce to 3-4 parameters maximum or get more data (240+ months). Not "no concern" (A), not "acceptable" (B - too few), not N² (D - that\'s extreme, would need 10,000+ months!). The 30× rule is the practical guideline.',
    },
    {
      id: 'bp-mc-3',
      type: 'multiple-choice' as const,
      question:
        'In-sample Sharpe 1.1, out-of-sample Sharpe 0.4. This indicates:',
      options: [
        'Good robust strategy',
        'Moderate overfitting',
        'Severe overfitting',
        'Strategy improved out-of-sample',
      ],
      correctAnswer: 2,
      explanation:
        "Answer: C. Sharpe collapsed 64% from in-sample to out-of-sample (1.1 → 0.4). This is severe overfitting. Good strategies degrade 10-20%, moderate overfitting 30-40%, severe >50%. The strategy learned noise, not signal. NOT robust (A), worse than moderate (B), definitely not improved (D). This backtest should be rejected - the strategy won't work in live trading.",
    },
    {
      id: 'bp-mc-4',
      type: 'multiple-choice' as const,
      question:
        'Walk-forward analysis with 5-year training, 1-year testing, over 24 years gives:',
      options: [
        '4 out-of-sample periods',
        '19 out-of-sample periods',
        '24 out-of-sample periods',
        '5 out-of-sample periods',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. Anchored walk-forward: Train on years 1-5, test year 6. Train on 1-6, test 7. ... Train on 1-23, test 24. That's 19 one-year OOS periods (years 6-24). NOT 4 (A - that would be non-overlapping blocks), not 24 (C - year 1-5 are training only), not 5 (D - incorrect count). Walk-forward provides 19 independent OOS tests, excellent validation methodology.",
    },
    {
      id: 'bp-mc-5',
      type: 'multiple-choice' as const,
      question: 'Survivorship bias in backtests typically causes:',
      options: [
        'Underestimation of returns by 0.5-1% annually',
        'Overestimation of returns by 1-3% annually',
        'No effect on returns, only on risk',
        'Correct returns but underestimated risk',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B. Survivorship bias includes only companies that survived, excluding bankruptcies. Overestimates returns 1-3% annually (small-cap more, large-cap less). Classic mistake: using current S&P 500 members retroactively. Must use point-in-time data with delistings. NOT underestimation (A), not no effect (C), not just risk (D - affects both returns and risk, but returns more). Survivorship bias makes backtests look unrealistically good.',
    },
  ],
};
