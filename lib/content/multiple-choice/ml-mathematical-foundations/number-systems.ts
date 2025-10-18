/**
 * Multiple choice questions for Number Systems & Properties section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const numbersystemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-number-systems',
    question:
      'Which number system is NOT closed under division (excluding division by zero)?',
    options: [
      'Rational numbers (ℚ)',
      'Real numbers (ℝ)',
      'Integers (ℤ)',
      'Complex numbers (ℂ)',
    ],
    correctAnswer: 2,
    explanation:
      'Integers are NOT closed under division. For example, 5 ÷ 2 = 2.5, which is not an integer. Rationals, reals, and complex numbers are all closed under division (except by zero).',
  },
  {
    id: 'mc2-floating-point',
    question:
      'What is the main reason why 0.1 + 0.2 != 0.3 in most programming languages?',
    options: [
      'Programming language bug',
      'Binary floating-point representation cannot exactly represent 0.1 and 0.2',
      'Insufficient memory allocation',
      'Integer overflow',
    ],
    correctAnswer: 1,
    explanation:
      'In binary floating-point representation (IEEE 754), decimal fractions like 0.1 and 0.2 cannot be represented exactly. They are approximated, leading to tiny rounding errors that accumulate. This is why 0.1 + 0.2 results in something like 0.30000000000000004.',
  },
  {
    id: 'mc3-scientific-notation',
    question:
      'In machine learning, a typical learning rate of 0.001 is best expressed in scientific notation as:',
    options: ['1 × 10^3', '1 × 10^-3', '10 × 10^-2', '0.1 × 10^-2'],
    correctAnswer: 1,
    explanation:
      '0.001 = 1 × 10^-3. In scientific notation, we express numbers as a × 10^n where 1 ≤ |a| < 10. While options C and D are technically correct, option B follows the standard scientific notation convention.',
  },
  {
    id: 'mc4-absolute-value',
    question:
      'The Triangle Inequality states that |x + y| ≤ |x| + |y|. Which of the following demonstrates when equality holds?',
    options: [
      'When x and y are both negative',
      'When x and y have the same sign',
      'When x and y have opposite signs',
      'When one of them is zero',
    ],
    correctAnswer: 1,
    explanation:
      'Equality holds when x and y have the same sign (both positive or both negative). When they have the same sign, there is no cancellation, so |x + y| = |x| + |y|. When they have opposite signs, some cancellation occurs, making |x + y| < |x| + |y|.',
  },
  {
    id: 'mc5-numerical-stability',
    question:
      'Why do we subtract the maximum value before computing softmax: exp(x - max(x)) / sum(exp(x - max(x)))?',
    options: [
      'To make the computation faster',
      'To prevent numerical overflow when exponentiating large values',
      'To ensure the output sums to 1',
      'To make all values negative',
    ],
    correctAnswer: 1,
    explanation:
      "Subtracting the maximum value before exponentiation prevents numerical overflow. exp(1000) would overflow, but exp(1000 - 1000) = exp(0) = 1 is manageable. This transformation doesn't change the final result due to properties of exponents but makes computation numerically stable.",
  },
];
