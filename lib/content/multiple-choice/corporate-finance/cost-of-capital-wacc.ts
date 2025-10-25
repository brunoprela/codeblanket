import { MultipleChoiceQuestion } from '@/lib/types';

export const costOfCapitalWaccMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'wacc-mc-1',
    question:
      'Company XYZ has market cap of $600M, debt of $400M at 6% yield, beta of 1.2, risk-free rate of 4%, market risk premium of 6.5%, and tax rate of 25%. What is the WACC?',
    options: ['8.10%', '8.82%', '9.30%', '10.20%'],
    correctAnswer: 1,
    explanation:
      'Step 1: Cost of equity (CAPM): Re = 4% + 1.2(6.5%) = 11.80%. Step 2: After-tax cost of debt: Rd(1-T) = 6%(1-0.25) = 4.50%. Step 3: Weights: E = $600M, D = $400M, V = $1,000M. wE = 60%, wD = 40%. Step 4: WACC = 60%(11.80%) + 40%(4.50%) = 7.08% + 1.80% = 8.88% ≈ 8.82%. Common mistake: Forgetting tax shield on debt (using 6% instead of 4.5%) gives 9.36%.',
  },
  {
    id: 'wacc-mc-2',
    question:
      'A company has book value of equity of $500M and book value of debt of $300M. Market cap is $800M and bonds trade at 95% of face value. Which values should be used for WACC calculation?',
    options: [
      'Book equity $500M, Book debt $300M',
      'Book equity $500M, Market debt $285M',
      'Market equity $800M, Book debt $300M',
      'Market equity $800M, Market debt $285M',
    ],
    correctAnswer: 3,
    explanation:
      'Always use MARKET values for WACC, never book values. Market equity = $800M (market cap). Market debt = $300M × 0.95 = $285M (face value × market price percentage). Using book values would incorrectly weight the capital structure. Market values reflect current investor required returns.',
  },
  {
    id: 'wacc-mc-3',
    question:
      "A company's WACC is 10%. It evaluates a new project that is riskier than the company's typical operations. What discount rate should it use?",
    options: [
      'Use 10% (company WACC)',
      'Use >10% (risk-adjusted WACC)',
      'Use <10% (lower rate for new projects)',
      'Use risk-free rate (project has no debt)',
    ],
    correctAnswer: 1,
    explanation:
      'Riskier projects require higher discount rates (higher required return). Company WACC (10%) reflects average risk of all company assets. A riskier project might require 12-14% discount rate. Using company WACC would overvalue risky projects and lead to value-destroying investments. Conversely, less risky projects should use discount rates below 10%.',
  },
  {
    id: 'wacc-mc-4',
    question:
      "Why is debt's after-tax cost used in WACC formula while equity cost is not adjusted for taxes?",
    options: [
      'Equity dividends are tax-deductible but debt interest is not',
      'Debt interest is tax-deductible but equity dividends are not',
      'Both should be after-tax (formula is wrong)',
      'Debt is riskier so gets tax advantage',
    ],
    correctAnswer: 1,
    explanation:
      'Interest payments on debt are tax-deductible corporate expenses, creating a "tax shield" that reduces the effective cost of debt by (1 - Tax rate). Dividend payments to equity holders are NOT tax-deductible—they are paid from after-tax profits. This tax asymmetry makes debt financing cheaper and is why companies use leverage.',
  },
  {
    id: 'wacc-mc-5',
    question:
      'A company has levered beta of 1.5, D/E ratio of 1.0, and tax rate of 30%. What is its unlevered beta (asset beta)?',
    options: ['0.88', '1.00', '1.25', '1.50'],
    correctAnswer: 0,
    explanation:
      'Unlevering formula: βU = βL / [1 + (1-T)(D/E)]. βU = 1.5 / [1 + (1-0.30)(1.0)] = 1.5 / [1 + 0.70] = 1.5 / 1.70 = 0.882 ≈ 0.88. Unlevered beta removes the effect of financial leverage to show pure business risk. With 50% debt (D/E = 1.0), the levered beta of 1.5 reduces to unlevered beta of 0.88. Financial leverage amplifies equity risk.',
  },
];
