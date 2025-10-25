import { MultipleChoiceQuestion } from '@/lib/types';

export const financialMathPuzzlesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fmp-mc-1',
    question:
      'What is the present value of $1,000 received in 3 years at 5% annual interest?',
    options: ['$850', '$863.84', '$900', '$950'],
    correctAnswer: 1,
    explanation:
      'PV = FV / (1+r)^n = 1000 / (1.05)^3 = 1000 / 1.157625 ≈ $863.84. Mental check: (1.05)^3 ≈ 1.16, so PV ≈ 1000/1.16 ≈ $862, which is close.',
  },
  {
    id: 'fmp-mc-2',
    question:
      'Using the Rule of 72, how long does it take to double money at 9% per year?',
    options: ['6 years', '8 years', '9 years', '12 years'],
    correctAnswer: 1,
    explanation:
      'Rule of 72: Years to double ≈ 72 / rate = 72 / 9 = 8 years. Exact: solve 2 = (1.09)^n → n = ln(2)/ln(1.09) = 0.693/0.086 ≈ 8.04 years. Rule of 72 is very accurate!',
  },
  {
    id: 'fmp-mc-3',
    question:
      'A perpetuity pays $100 annually. At 4% discount rate, what is its value?',
    options: ['$400', '$2,000', '$2,500', '$4,000'],
    correctAnswer: 2,
    explanation:
      'Perpetuity value = C / r = 100 / 0.04 = $2,500. This is the present value of infinite stream of $100 payments discounted at 4%.',
  },
  {
    id: 'fmp-mc-4',
    question:
      'Stock pays $3 dividend, growing at 5% annually. Required return is 10%. Stock value by Gordon model?',
    options: ['$30', '$45', '$60', '$75'],
    correctAnswer: 2,
    explanation:
      'Gordon Growth Model: P = D / (r - g) = 3 / (0.10 - 0.05) = 3 / 0.05 = $60. Only valid when g < r (growth rate less than required return).',
  },
  {
    id: 'fmp-mc-5',
    question:
      'Forward price for stock at $50, risk-free rate 4%, dividend yield 2%, T=1 year?',
    options: ['$49', '$51', '$52', '$54'],
    correctAnswer: 1,
    explanation:
      'F = S × e^((r-q)T) = 50 × e^(0.02) ≈ 50 × 1.0202 ≈ $51.01. The forward price reflects the risk-free rate minus dividend yield (cost of carry).',
  },
];
