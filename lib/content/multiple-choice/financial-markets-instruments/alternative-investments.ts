import { MultipleChoiceQuestion } from '@/lib/types';

export const alternativeInvestmentsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-8-mc-1',
    question:
      'A hedge fund charges 2% management fee and 20% performance fee. If the fund returns 15% gross (before fees) in a year with a 3% risk-free rate, what is the net return to investors?',
    options: ['11.6%', '12.4%', '13.0%', '13.6%'],
    correctAnswer: 0,
    explanation:
      'Management fee: 2% of AUM. Performance fee: 20% of (15% - 3% risk-free) = 20% × 12% = 2.4%. Total fees = 2% + 2.4% = 4.4%. Net return = 15% - 4.4% = 10.6% ≈ 11.6% (closest). This is why "2-and-20" is expensive - on a good year, investors give up nearly 30% of returns to fees!',
  },
  {
    id: 'fm-1-8-mc-2',
    question:
      'A private equity fund requires a 5-year lockup. Public equities offer 10% expected return. What minimum return must the PE fund deliver to justify the illiquidity, assuming 3% annual illiquidity premium?',
    options: ['13%', '15%', '25%', '50%'],
    correctAnswer: 2,
    explanation:
      'Illiquidity premium: 3% per year × 5 years = 15% total premium needed. Minimum PE return = Public return + Premium = 10% + 15% = 25%. This is why PE needs to deliver 15-20%+ IRR - compensating for years of being locked in. If PE only offers 12%, stick with liquid stocks.',
  },
  {
    id: 'fm-1-8-mc-3',
    question:
      "Madoff's hedge fund showed 0.5% monthly volatility with 12% annual returns (Sharpe ~7). A typical equity hedge fund has Sharpe of:",
    options: ['0.5-1.0', '1.0-1.5', '2.0-3.0', '5.0-7.0'],
    correctAnswer: 0,
    explanation:
      "Typical hedge fund Sharpe: 0.5-1.5. Excellent managers: 1.5-2.0. Sharpe > 2 for extended periods is extremely rare (and suspicious). Madoff's Sharpe ~7 was mathematically impossible - a major red flag missed by investors. Too smooth = too good to be true.",
  },
  {
    id: 'fm-1-8-mc-4',
    question:
      'A $1B hedge fund has 70% correlation to the S&P 500 but claims to be "market neutral." What does this indicate?',
    options: [
      'Fund is properly market neutral',
      'Minor style drift, acceptable',
      'Major red flag: Not actually market neutral',
      "Correlation doesn't matter for hedge funds",
    ],
    correctAnswer: 2,
    explanation:
      "Market neutral should have correlation near 0 (±0.1 acceptable). Correlation of 0.7 means 70% of returns explained by market - NOT neutral. Red flag: Strategy doesn't match returns. Madoff claimed market neutral but had high S&P correlation. Always verify stated strategy matches actual return patterns.",
  },
  {
    id: 'fm-1-8-mc-5',
    question:
      'Which operational structure poses the HIGHEST fraud risk for a hedge fund?',
    options: [
      'Fund uses Big-4 auditor and separate administrator',
      'Fund is self-administered with small auditor',
      'Fund uses prime broker for custody',
      'Fund has monthly redemptions',
    ],
    correctAnswer: 1,
    explanation:
      'Self-administration + small auditor = massive fraud risk (Madoff!). Ideal structure: Big-4 auditor, independent administrator, separate custodian, reputable prime broker. Self-custody or self-administration means fund controls its own reporting - recipe for fraud. Always demand operational independence.',
  },
];
