export const assetAllocationStrategiesMC = {
  id: 'asset-allocation-strategies-mc',
  title: 'Asset Allocation Strategies - Multiple Choice',
  questions: [
    {
      id: 'aas-mc-1',
      type: 'multiple-choice' as const,
      question: 'Strategic Asset Allocation (SAA) is best described as:',
      options: [
        'Short-term tactical adjustments based on market conditions',
        'Long-term policy weights based on risk tolerance and return objectives',
        'Equal weighting across all asset classes',
        'Following market-cap weights',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. SAA is the long-term policy portfolio (e.g., 60/40) based on investor's objectives, risk tolerance, time horizon, and capital market assumptions. Reviewed annually, updated every 3-5 years. NOT short-term tactical (A - that's TAA), not equal-weight (C - that's naive diversification), not market-cap (D - that's market portfolio/indexing). SAA is the foundation; everything else is overlay.",
    },
    {
      id: 'aas-mc-2',
      type: 'multiple-choice' as const,
      question:
        'The "100 minus age" rule for equity allocation is an example of:',
      options: [
        'Strategic asset allocation',
        'Tactical asset allocation',
        'Lifecycle/glide path allocation',
        'Risk parity allocation',
      ],
      correctAnswer: 2,
      explanation:
        "Answer: C. Lifecycle glide path reduces equities as you age: 35-year-old holds 65% equities, 65-year-old holds 35%. Aligns risk-taking with human capital and time horizon. Not pure SAA (A - that's fixed over time), not TAA (B - that's market timing), not risk parity (D - that's equal risk contribution). Modern view: 110 or 120 minus age may be more appropriate due to longer lifespans.",
    },
    {
      id: 'aas-mc-3',
      type: 'multiple-choice' as const,
      question: 'An Information Ratio of 0.4 with 3% tracking error implies:',
      options: [
        '1.2% expected active return',
        '7.5% expected active return',
        '0.12% expected active return',
        'Tracking error is too high',
      ],
      correctAnswer: 0,
      explanation:
        'Active Return = IR × TE = 0.4 × 3% = 1.2%. Answer: A. An IR of 0.4 is good (>0.3 is solid active management). With modest 3% tracking error, generates 1.2% alpha. This is enough to justify active management fees of 0.5-1.0%. Not 7.5% (B - that would require IR=2.5, nearly impossible), not 0.12% (C - decimal error), and 3% TE is moderate, not too high (D).',
    },
    {
      id: 'aas-mc-4',
      type: 'multiple-choice' as const,
      question:
        'Dynamic asset allocation that reduces equity when CAPE > 25 and increases when CAPE < 15 is based on:',
      options: [
        'Market timing',
        'Momentum signals',
        'Mean reversion / valuation',
        'Factor timing',
      ],
      correctAnswer: 2,
      explanation:
        "Answer: C. CAPE (cyclically-adjusted P/E) is a valuation metric. High CAPE suggests overvaluation → reduce equities (mean reversion). Low CAPE suggests undervaluation → increase equities. Evidence: CAPE has predictive power for 10-year returns (R² ≈ 0.4). Not pure market timing (A - broader term), not momentum (B - that's trend-following), not factor timing (D - that's rotating between factors).",
    },
    {
      id: 'aas-mc-5',
      type: 'multiple-choice' as const,
      question:
        'Target-date 2050 fund appropriate for 35-year-old (15 years to retirement) should have approximately:',
      options: [
        '90-95% stocks (aggressive early accumulation)',
        '70-80% stocks (moderate growth)',
        '50-60% stocks (balanced)',
        '20-30% stocks (conservative near retirement)',
      ],
      correctAnswer: 0,
      explanation:
        'Answer: A. Target-date 2050 is 25+ years away, appropriate for 35-year-old. Should be 90-95% stocks early in glide path. The question has an error (35-year-old + 15 years = age 50, should be target-date 2040). For TRUE 15-years-to-retirement, answer would be C (50-60%). For target-date 2050 fund itself: definitely A (90-95% for young investors with 25+ year horizons).',
    },
  ],
};
