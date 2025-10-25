/**
 * Multiple choice questions for Sequences & Series section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const sequencesseriesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-arithmetic-sequence',
    question:
      'What is the 50th term of the arithmetic sequence 5, 9, 13, 17, ...?',
    options: ['201', '205', '197', '209'],
    correctAnswer: 0,
    explanation:
      'Formula: aₙ = a₁ + (n-1)d where a₁=5, d=4, n=50. Therefore a₅₀ = 5 + (50-1)×4 = 5 + 49×4 = 5 + 196 = 201.',
  },
  {
    id: 'mc2-geometric-series',
    question:
      'What is the sum of the infinite geometric series 1 + 1/3 + 1/9 + 1/27 + ...?',
    options: ['1', '1.5', '2', '3'],
    correctAnswer: 1,
    explanation:
      'This is a geometric series with a₁=1 and r=1/3. Since |r|<1, it converges. Sum = a₁/(1-r) = 1/(1-1/3) = 1/(2/3) = 3/2 = 1.5.',
  },
  {
    id: 'mc3-fibonacci',
    question:
      'If the Fibonacci sequence is 1, 1, 2, 3, 5, 8, 13, ..., what is the 10th term?',
    options: ['34', '55', '89', '21'],
    correctAnswer: 1,
    explanation:
      'Continue the sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55. The 10th term is 55. Each term is the sum of the previous two.',
  },
  {
    id: 'mc4-summation',
    question: 'Evaluate: Σ(i=1 to 100) i²',
    options: ['5,050', '338,350', '250,000', '1,000,000'],
    correctAnswer: 1,
    explanation:
      'Formula for sum of squares: Σi² = n (n+1)(2n+1)/6. For n=100: 100×101×201/6 = 2,030,100/6 = 338,350.',
  },
  {
    id: 'mc5-convergence',
    question: 'Which sequence converges?',
    options: ['aₙ = n', 'aₙ = (-1)ⁿ', 'aₙ = 1/n', 'aₙ = n²'],
    correctAnswer: 2,
    explanation:
      'aₙ = 1/n converges to 0 as n→∞. The others: n→∞ (diverges), (-1)ⁿ oscillates (no limit), n²→∞ (diverges).',
  },
];
