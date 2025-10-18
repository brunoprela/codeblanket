/**
 * Multiple choice questions for Combinatorics Basics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const combinatoricsbasicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1-combinations',
    question:
      'You have 10 features and want to select exactly 3 for your model. How many different feature sets are possible?',
    options: ['30', '120', '720', '1000'],
    correctAnswer: 1,
    explanation:
      "This is C(10, 3) = 10!/(3!×7!) = (10×9×8)/(3×2×1) = 720/6 = 120. Order doesn't matter for feature selection, so we use combinations.",
  },
  {
    id: 'mc2-permutations',
    question:
      'In how many ways can you arrange 4 different models in an ensemble pipeline where order matters?',
    options: ['4', '16', '24', '256'],
    correctAnswer: 2,
    explanation:
      'This is P(4, 4) = 4! = 4×3×2×1 = 24. Order matters, so we use permutations of all 4 models.',
  },
  {
    id: 'mc3-grid-search',
    question:
      'A grid search has 3 learning rates, 4 batch sizes, and 2 optimizers. How many total configurations?',
    options: ['9', '12', '24', '64'],
    correctAnswer: 2,
    explanation:
      'Use multiplication principle: 3 × 4 × 2 = 24 total configurations.',
  },
  {
    id: 'mc4-binomial-coefficient',
    question: 'What is C(6, 2)?',
    options: ['12', '15', '30', '720'],
    correctAnswer: 1,
    explanation: 'C(6, 2) = 6!/(2!×4!) = (6×5)/(2×1) = 30/2 = 15.',
  },
  {
    id: 'mc5-pascals-triangle',
    question: "What is the sum of all numbers in row n of Pascal's triangle?",
    options: ['n', 'n!', '2^n', 'n^2'],
    correctAnswer: 2,
    explanation:
      "The sum of row n in Pascal's triangle equals 2^n. This is because C(n,0) + C(n,1) + ... + C(n,n) = 2^n.",
  },
];
