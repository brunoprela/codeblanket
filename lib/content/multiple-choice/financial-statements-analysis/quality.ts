export const qualityMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'A company has CFO of $80M, Net Income of $100M, and Accruals of $20M. What is the CFO/NI ratio and what does it indicate?',
    options: [
      '1.25 - Excellent earnings quality',
      '0.80 - Concerning; only 80% of earnings converted to cash, suggests potential earnings inflation through accruals',
      '1.20 - Above benchmark, high quality',
      'The ratio cannot be calculated from this data',
      '0.80 - Normal for growing companies',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. CFO/NI = $80M / $100M = 0.80. This means only 80% of reported earnings converted to actual cash. The $20M gap is accruals (non-cash earnings). While some accruals are normal, a ratio <1.0 is concerning because: (1) Company reporting $100M profit but only generated $80M cash, (2) $20M accruals could be aggressive revenue recognition or understated expenses, (3) Research shows companies with low CFO/NI ratios underperform subsequently. Benchmark is >1.0 for high-quality earnings. This warrants investigation into working capital and accrual quality.`,
  },

  {
    id: 2,
    question: "Company's Beneish M-Score is -1.78. What does this indicate?",
    options: [
      'Low manipulation risk (score below -2.22 threshold)',
      'High manipulation risk; M-Score > -2.22 suggests probable earnings manipulation - investigate DSRI, GMI, and accruals components',
      'The company is bankrupt',
      'Average earnings quality with no concerns',
      'Score is too low to be meaningful',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Beneish M-Score > -2.22 indicates HIGH probability of earnings manipulation. Score of -1.78 is ABOVE the threshold, suggesting the company exhibits characteristics common to known manipulators (Enron, WorldCom had scores >-2.22). This doesn't prove fraud but means: (1) Multiple financial ratios show manipulation patterns, (2) Warrants deep investigation of revenue recognition, accruals, margins, (3) Consider avoiding investment. Research shows M-Score >-2.22 predicts 76% of manipulators correctly. Note: More negative is better (e.g., -3.0 is safer than -1.5).`,
  },

  {
    id: 3,
    question: 'Altman Z-Score of 1.5. What does this indicate?',
    options: [
      'Safe zone - no bankruptcy risk',
      'Grey zone - uncertain outlook',
      'Distress zone (Z < 1.81) - high bankruptcy risk within 2 years; company likely struggling with leverage, profitability, or liquidity',
      'Score indicates strong financial health',
      'Z-Score is not applicable to this company',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Altman Z-Score of 1.5 is in the DISTRESS ZONE (Z < 1.81), indicating high probability of bankruptcy within 1-2 years. The score combines: (1) Working capital, (2) Retained earnings, (3) EBIT, (4) Market value vs liabilities, (5) Asset turnover. A score of 1.5 suggests: Company has negative working capital or high leverage, poor profitability, low market cap relative to debt, or asset inefficiency. Historical accuracy: 80-90% for predicting bankruptcy. Action: AVOID investment, or if holding, consider selling. Note: Safe zone is Z > 2.99, Grey zone is 1.81-2.99.`,
  },

  {
    id: 4,
    question: 'Piotroski F-Score of 3 out of 9. What does this mean?',
    options: [
      'High quality value stock - should buy',
      'Medium quality - neutral signal',
      'Low quality (F-Score ≤3) - weak fundamentals across profitability, leverage, and efficiency; avoid despite low valuation',
      'Score is too high to interpret',
      'Company is bankrupt',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Piotroski F-Score of 3/9 is LOW QUALITY. The 9-point scorecard evaluates: Profitability (4 pts), Leverage/Liquidity (3 pts), Operating Efficiency (2 pts). Score of 3 means company only passes 3 tests, failing 6. This suggests: (1) Likely negative or declining profitability, (2) Worsening leverage or liquidity, (3) Poor operating efficiency. Research by Piotroski shows: F-Score ≥7 stocks outperform by 7.5%/year, F-Score ≤3 stocks underperform significantly. Even if valuation looks cheap (low P/B), fundamental weakness makes it a value trap. Action: AVOID - low quality rarely improves.`,
  },

  {
    id: 5,
    question:
      'AR grew 30%, revenue grew 10%, inventory grew 25%. What is most likely happening?',
    options: [
      'Normal growth pattern for expanding company',
      'Strong sales demand driving inventory buildup',
      "Likely channel stuffing: Company pushing product to distributors (AR up) who aren't selling it (inventory builds up) to inflate current period revenue",
      'Efficient working capital management',
      'Seasonal business cycle effects',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. This is a textbook channel stuffing pattern: (1) AR growing 3x faster than revenue (30% vs 10%) means customers not paying or extended credit terms to boost sales, (2) Inventory growing 2.5x faster means products accumulating unsold, (3) Together: Company likely stuffed distribution channel with excess product using liberal credit terms to meet revenue targets. Result: Current quarter looks good, but: Next quarter faces difficult comps, AR may become uncollectible, Inventory write-downs likely, Revenue reverses when channel clears. Historical examples: Luckin Coffee, Autonomy (HP), Sunbeam. Action: Calculate DSO and DIO trends, check if this is persistent, likely AVOID or SHORT.`,
  },
];
