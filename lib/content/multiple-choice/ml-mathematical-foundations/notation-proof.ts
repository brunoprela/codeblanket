/**
 * Multiple choice questions for Mathematical Notation & Proof section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const notationproofMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-summation',
    question: 'What does Σ(i=1 to 5) i² equal?',
    options: ['15', '25', '55', '225'],
    correctAnswer: 2,
    explanation:
      'Σ(i=1 to 5) i² = 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55. Or use formula: n (n+1)(2n+1)/6 = 5×6×11/6 = 55.',
  },
  {
    id: 'mc2-quantifiers',
    question: '∀x ∈ ℝ, x² ≥ 0 means:',
    options: [
      'Some real numbers have non-negative squares',
      'All real numbers have non-negative squares',
      'There exists a real number with negative square',
      'No real numbers have non-negative squares',
    ],
    correctAnswer: 1,
    explanation:
      '∀ means "for all". The statement says for all real numbers x, x² is non-negative, which is true.',
  },
  {
    id: 'mc3-set-notation',
    question: 'If A = {1, 2, 3} and B = {2, 3, 4}, what is A ∩ B?',
    options: ['{1}', '{2, 3}', '{1, 2, 3, 4}', '{4}'],
    correctAnswer: 1,
    explanation:
      'A ∩ B (intersection) contains elements in both A and B: {2, 3}.',
  },
  {
    id: 'mc4-function-composition',
    question: 'If f (x) = 2x and g (x) = x + 1, what is (f ∘ g)(3)?',
    options: ['7', '8', '10', '14'],
    correctAnswer: 1,
    explanation:
      '(f ∘ g)(3) means f (g(3)). First g(3) = 3+1 = 4, then f(4) = 2×4 = 8.',
  },
  {
    id: 'mc5-product-notation',
    question: 'What is Π(i=1 to 4) i (product of first 4 positive integers)?',
    options: ['4', '10', '16', '24'],
    correctAnswer: 3,
    explanation: 'Π(i=1 to 4) i = 1 × 2 × 3 × 4 = 24 = 4!',
  },
];
