export const factorInvestingStrategiesMC = [
  {
    id: 'ats-11-mc-1',
    question: 'Fama-French 3-factor model includes:',
    options: [
      'Market, Value, Momentum',
      'Market, Size, Value',
      'Market, Momentum, Quality',
      'Value, Momentum, Size',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Market, Size, Value. Fama-French (1993) model: Market (beta), Size (SMB = Small Minus Big), Value (HML = High Minus Low B/P). Later extended to 5-factor (added Profitability, Investment).',
  },
  {
    id: 'ats-11-mc-2',
    question:
      'Multi-factor portfolio with 5 uncorrelated factors (each Sharpe 0.6) has combined Sharpe:',
    options: [
      '0.6 (same)',
      '1.34 (√5 × 0.6)',
      '3.0 (5 × 0.6)',
      '0.27 (0.6 / √5)',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: 1.34 (√5 × 0.6). With uncorrelated factors, Sharpe scales as √N. Five factors: √5 × 0.6 = 2.24 × 0.6 = 1.34. Diversification benefit. If correlated (ρ=0.3), Sharpe ≈ 1.05.',
  },
  {
    id: 'ats-11-mc-3',
    question: 'Factor crowding is dangerous because:',
    options: [
      'Factors stop working permanently',
      'Synchronized selling causes crashes',
      'Correlations decrease',
      'Spreads widen',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: Synchronized selling causes crashes. When factor crowded, many funds hold same positions. Factor reversal → everyone sells simultaneously → crash. Example: 2020 value crash (-20% in 1 month).',
  },
  {
    id: 'ats-11-mc-4',
    question: 'Value factor (low P/E, high B/P) historically provides:',
    options: [
      'No premium',
      '2-3% annual premium',
      '10%+ annual premium',
      'Negative premium',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: 2-3% annual premium. Fama-French value premium (HML) ≈ 3-4% annual (1927-2024). Varies by period. 2010-2020: value underperformed (growth dominated). 2022+: value outperformed (rates rose).',
  },
  {
    id: 'ats-11-mc-5',
    question: 'Factor investing fees typically range:',
    options: [
      '0.03-0.10% (like index funds)',
      '0.20-0.50% (smart beta ETFs)',
      '1.00-2.00% (like hedge funds)',
      '2.00%+ (very expensive)',
    ],
    correctAnswer: 1,
    explanation:
      'Correct: 0.20-0.50% (smart beta ETFs). Factor ETFs (AQR, iShares) charge 0.2-0.5%. More than passive (0.03-0.1%) but far less than active funds (1-2%). Justifiable for systematic alpha.',
  },
];
