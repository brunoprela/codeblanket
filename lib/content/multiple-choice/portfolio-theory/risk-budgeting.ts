export const riskBudgetingMC = {
  id: 'risk-budgeting-mc',
  title: 'Risk Budgeting - Multiple Choice',
  questions: [
    {
      id: 'rb-mc-1',
      type: 'multiple-choice' as const,
      question: 'Risk parity portfolio construction aims for:',
      options: [
        'Equal weights across all assets',
        'Equal risk contribution from each asset',
        'Minimum total portfolio risk',
        'Maximum Sharpe ratio',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. Risk parity equalizes marginal risk contribution: each asset contributes 1/N of total portfolio risk. Bonds get high weights (low vol), stocks get low weights (high vol). NOT equal weights (A - that's naive diversification), not min risk (C - that's GMVP), not max Sharpe (D - that's tangency portfolio). Risk parity typically requires leverage to achieve equity-like returns.",
    },
    {
      id: 'rb-mc-2',
      type: 'multiple-choice' as const,
      question:
        'For 4-asset risk parity portfolio, each asset should contribute:',
      options: [
        '25% of total portfolio return',
        '25% of total portfolio volatility',
        '25% of total portfolio variance',
        '25% of total portfolio risk (measured by volatility contribution)',
      ],
      correctAnswer: 3,
      explanation:
        'Answer: D. Equal risk contribution means each asset contributes 25% of portfolio risk. Marginal contribution: MCR_i = w_i × (Σw)_i / σ_p. For 4 assets: MCR_1 = MCR_2 = MCR_3 = MCR_4 = σ_p/4. NOT return (A - returns differ), not simple volatility (B - that doesn\'t make sense dimensionally), not variance (C - we measure in volatility units). The "D" formulation is correct: 25% of portfolio risk.',
    },
    {
      id: 'rb-mc-3',
      type: 'multiple-choice' as const,
      question: 'Unlevered risk parity portfolio typically has:',
      options: [
        'Higher returns than 60/40, lower risk',
        'Similar returns to 60/40, lower risk',
        'Lower returns than 60/40, much lower risk',
        'Same risk as 60/40, higher returns',
      ],
      correctAnswer: 2,
      explanation:
        'Answer: C. Unlevered RP: ~9% vol, 6% return (bond-heavy). 60/40: ~11% vol, 7.5% return. RP has LOWER returns and LOWER risk due to heavy bond allocation (often 50-70% bonds). To match 60/40 risk, need 1.2-1.4x leverage, costing 0.5-1% in borrowing costs. Levered RP can match 60/40 returns with similar risk, but unlevered underperforms on returns (though better downside protection).',
    },
    {
      id: 'rb-mc-4',
      type: 'multiple-choice' as const,
      question:
        'Custom risk budget allocating 60% risk to equities, 40% to bonds (from equal weights) implies:',
      options: [
        'Increasing equity allocation',
        'Decreasing equity allocation',
        'No change needed',
        'Cannot determine without volatilities',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B (typically). Equal weights (50/50) give ~80-85% risk from equities (higher volatility). Target 60% equity risk requires DECREASING equity weight to ~40-45%. Counterintuitive: reduce equity allocation to reduce its risk dominance. Answer D is tempting, but qualitatively, moving from equal-weight toward custom budget with lower equity risk % requires reducing equity allocation.',
    },
    {
      id: 'rb-mc-5',
      type: 'multiple-choice' as const,
      question:
        'Factor risk budgeting is superior to asset risk budgeting because:',
      options: [
        "It's computationally faster",
        'Factors are the fundamental risk drivers, more stable than asset correlations',
        'It requires fewer parameters',
        'It guarantees higher returns',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. Factors (equity, rates, credit, currency) are fundamental risk sources. Factor correlations more stable than asset correlations. Multi-asset portfolios naturally have factor concentration. Asset-level budgeting can miss factor concentration (e.g., stocks+credit+REITs all have high equity factor exposure). Not faster (A - more complex), not fewer parameters (C - similar or more), doesn't guarantee returns (D - nothing does). The insight: budget true risk factors, not asset proxies.",
    },
  ],
};
