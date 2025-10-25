export const creditMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'Company has EBIT of $200M and interest expense of $60M. What is the interest coverage ratio and what does it indicate?',
    options: [
      '3.33x - Strong coverage, investment grade',
      '3.33x - Adequate but below 4x benchmark; company can service debt but limited cushion, likely BBB/BB rating',
      '0.30x - Company cannot service debt',
      '5.0x - Excellent coverage',
      'Cannot be calculated',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Interest Coverage = EBIT / Interest = $200M / $60M = 3.33x. This means EBIT is 3.33 times interest expense - company can pay interest 3.33 times over. Interpretation: (1) >8x = AA/AAA (excellent), (2) 4-8x = A/BBB (solid investment grade), (3) 2.5-4x = BB/B (high yield), (4) <2.5x = distressed. At 3.33x, this is borderline investment grade/high yield, suggesting BBB or BB rating. Adequate to service debt but offers limited cushion if earnings decline. A 30% EBIT drop would bring coverage to 2.3x (concerning).`,
  },

  {
    id: 2,
    question:
      'Debt/EBITDA is 4.5x. Is this high or low leverage? What rating does it suggest?',
    options: [
      'Low leverage - investment grade',
      'Moderate leverage - A/BBB rating',
      'High leverage (>4x for corporates); suggests BB/B high yield rating with elevated default risk',
      'Distressed - near default',
      'Normal leverage for all industries',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Debt/EBITDA of 4.5x is HIGH leverage for most corporates. Benchmarks: (1) <2x = AA/AAA, (2) 2-3x = A/BBB (investment grade), (3) 3-5x = BB/B (high yield), (4) >5x = CCC/distressed. At 4.5x, company's debt is 4.5 times annual EBITDA - would take 4.5 years of ALL EBITDA to pay off debt (unrealistic). This suggests: BB or B credit rating, 300-500bps spread over Treasuries, vulnerable in downturn. Note: Some industries (utilities, REITs) tolerate higher leverage due to stable cash flows, but 4.5x is generally high yield territory.`,
  },

  {
    id: 3,
    question: 'Which debt position has highest recovery rate in default?',
    options: [
      'Subordinated unsecured bonds',
      'Senior unsecured bonds',
      'Senior secured bonds (typically 60-70% recovery vs 30-50% for senior unsecured, 15-30% for subordinated)',
      'Equity',
      'All have same recovery',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. In bankruptcy, recovery rates follow seniority: (1) Senior Secured: 60-70% (first claim on collateral), (2) Senior Unsecured: 30-50% (claim on general assets), (3) Subordinated: 15-30% (paid after senior debt), (4) Equity: 0-5% (residual claim). Senior secured bonds are backed by specific collateral (buildings, equipment) giving first priority. If company worth $100M and has $150M senior secured debt, secured holders get $100M (67% recovery). Unsecured get nothing. This is why secured bonds trade at lower spreads (50-100bps less) than unsecured at same company.`,
  },

  {
    id: 4,
    question:
      'Credit spread is 250bps. What does this tell you about the bond?',
    options: [
      'AAA rated - minimal risk',
      'Investment grade with low risk',
      'BB/B high yield rating; spread of 250bps indicates non-investment grade with moderate default risk',
      'Near default',
      'Cannot determine rating from spread alone',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Credit spread of 250bps over Treasuries indicates high yield (non-investment grade) rating. Typical spreads: (1) AAA/AA: <100bps, (2) A: 100-150bps, (3) BBB: 150-250bps, (4) BB: 250-400bps, (5) B: 400-700bps, (6) CCC: >700bps. At 250bps, likely BBB (low investment grade) or BB (high yield). The 250bp spread compensates investors for: elevated default risk (~2-5% over 5 years), lower recovery (40-60%), and liquidity risk. If you buy this bond yielding Treasury + 250bps, you're taking meaningful credit risk but not distressed-level risk.`,
  },

  {
    id: 5,
    question:
      'Covenant requires Debt/EBITDA < 4.0x. Company currently at 3.8x. How concerning is this?',
    options: [
      'No concern - well within covenant',
      'Very concerning - near breach; only 0.2x cushion means 5% EBITDA decline or small debt increase triggers violation, leading to default acceleration',
      'Moderate concern but manageable',
      "Covenants don't matter for bondholders",
      'Company should immediately refinance',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Being at 3.8x with 4.0x covenant max is VERY CONCERNING despite technically complying. Here's why: (1) Only 0.2x cushion = 5% EBITDA decline triggers breach, (2) Covenant breach = technical default, (3) Default = debt becomes immediately due (acceleration), (4) Even if waived, company has no negotiating leverage. Example: If EBITDA falls from $500M to $475M (5% drop), leverage goes to 4.0x = breach. Lenders can: demand immediate repayment, charge waiver fees, tighten covenants further, or force asset sales. Bondholders should: demand debt paydown, restrict new borrowing, or sell bonds. Cushion <10% of covenant threshold is "red zone".`,
  },
];
