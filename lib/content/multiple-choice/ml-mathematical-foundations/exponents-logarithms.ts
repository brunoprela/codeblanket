/**
 * Multiple choice questions for Exponents & Logarithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const exponentslogarithmsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-exponent-laws',
    question: 'Simplify: (2³)² · 2⁴ / 2⁵',
    options: ['2⁵', '2⁷', '2⁹', '2¹¹'],
    correctAnswer: 0,
    explanation:
      'Step by step: (2³)² = 2⁶ (power rule). Then 2⁶ · 2⁴ = 2¹⁰ (product rule). Finally 2¹⁰ / 2⁵ = 2⁵ (quotient rule). Answer: 2⁵ = 32.',
  },
  {
    id: 'mc2-logarithm-properties',
    question: 'If log₂(x) = 5, what is x?',
    options: ['10', '25', '32', '64'],
    correctAnswer: 2,
    explanation:
      'log₂(x) = 5 means 2⁵ = x. Therefore x = 32. Logarithms and exponents are inverse operations.',
  },
  {
    id: 'mc3-log-laws',
    question: 'Simplify: log(100) + log(10) - log(10)',
    options: ['log(100)', 'log(1000)', '2', '3'],
    correctAnswer: 0,
    explanation:
      'Using log laws: log(100) + log(10) = log(100·10) = log(1000). Then log(1000) - log(10) = log(1000/10) = log(100). If using base 10: log₁₀(100) = 2.',
  },
  {
    id: 'mc4-entropy',
    question:
      'A fair coin flip has entropy H = 1 bit. What is the entropy of a fair 4-sided die?',
    options: ['1 bit', '2 bits', '3 bits', '4 bits'],
    correctAnswer: 1,
    explanation:
      'For uniform distribution over n outcomes: H = log₂(n). For 4-sided die: H = log₂(4) = 2 bits. You need 2 bits to represent 4 equally likely outcomes.',
  },
  {
    id: 'mc5-compound-interest',
    question:
      'Which gives higher returns after 1 year: $100 at 12% compounded monthly, or $100 at 12% simple interest?',
    options: [
      'Simple interest',
      'Compound interest',
      'They are equal',
      'Cannot determine',
    ],
    correctAnswer: 1,
    explanation:
      'Simple: $100 + $100(0.12) = $112. Compound monthly: $100(1 + 0.12/12)¹² = $100(1.01)¹² ≈ $112.68. Compound interest is always higher than simple interest for the same nominal rate.',
  },
];
