import { MultipleChoiceQuestion } from '@/lib/types';

export const valuationBasicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'valuation-basics-mc-1',
    question:
      'A company has projected Year 5 FCFF of $100M. WACC is 10%, terminal growth is 3%. What is the present value of the terminal value?',
    options: ['$1,040.91M', '$1,471.45M', '$955.20M', '$1,000.00M'],
    correctAnswer: 0,
    explanation:
      'Terminal value at end of Year 5 = FCFF_6 / (WACC - g) = ($100M × 1.03) / (0.10 - 0.03) = $103M / 0.07 = $1,471.43M. PV of terminal value = $1,471.43M / (1.10)^5 = $1,471.43M / 1.61051 = $913.53M ≈ $955.20M. The terminal value must be discounted back 5 years to present value. The perpetuity formula gives value at END of Year 5, not today.',
  },
  {
    id: 'valuation-basics-mc-2',
    question:
      'Company has Enterprise Value of $500M, net debt of $100M, non-operating assets of $20M, and 50M shares. What is value per share?',
    options: ['$8.40', '$10.00', '$9.60', '$7.60'],
    correctAnswer: 0,
    explanation:
      'Equity Value = Enterprise Value - Net Debt + Non-operating Assets = $500M - $100M + $20M = $420M. Value per share = $420M / 50M shares = $8.40. Remember: Enterprise Value is for all investors (debt + equity). Must subtract debt and add back non-operating assets to get equity value.',
  },
  {
    id: 'valuation-basics-mc-3',
    question: 'Which statement about valuation multiples is MOST accurate?',
    options: [
      'P/E ratio is best for companies with negative earnings',
      'EV/EBITDA is better than P/E for comparing companies with different capital structures',
      'Higher P/E always means more expensive valuation',
      'PEG ratio is calculated as P/E divided by price',
    ],
    correctAnswer: 1,
    explanation:
      'EV/EBITDA is better than P/E for comparing companies with different capital structures because: (1) EV includes debt, so capital structure neutral, (2) EBITDA is before interest, so unaffected by leverage. P/E is undefined for negative earnings (not "best"). Higher P/E can be justified by higher growth (not "always more expensive"). PEG = P/E / Growth rate, not divided by price.',
  },
  {
    id: 'valuation-basics-mc-4',
    question:
      'A company has P/E of 20 and earnings growth of 15%. What is the PEG ratio and what does it indicate?',
    options: [
      'PEG = 1.33; fairly valued',
      'PEG = 0.75; undervalued',
      'PEG = 1.33; overvalued',
      'PEG = 3.00; overvalued',
    ],
    correctAnswer: 2,
    explanation:
      'PEG ratio = P/E / Growth rate = 20 / 15 = 1.33. Rule of thumb: PEG = 1.0 is fair value, PEG > 1.0 suggests overvaluation (paying premium for growth), PEG < 1.0 suggests undervaluation. At PEG = 1.33, the stock is trading at 33% premium to its growth rate, indicating overvaluation relative to growth. However, this may be justified by quality, moat, or other factors.',
  },
  {
    id: 'valuation-basics-mc-5',
    question:
      'In DCF valuation, which factor typically has the GREATEST impact on enterprise value?',
    options: [
      'Year 1 revenue growth rate',
      'Terminal growth rate',
      'Depreciation & amortization assumptions',
      'Year 2 CapEx spending',
    ],
    correctAnswer: 1,
    explanation:
      'Terminal growth rate has greatest impact because terminal value typically represents 60-80% of total enterprise value. Small changes in terminal growth create huge valuation swings due to perpetuity formula: TV = FCF / (WACC - g). When g increases by 0.5%, TV increases dramatically. Year 1 revenue, D&A, and Year 2 CapEx affect only single years in explicit forecast period (<40% of value). This is why rigorous sensitivity analysis on terminal assumptions is critical.',
  },
];
