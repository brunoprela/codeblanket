/**
 * Multiple choice questions for Combinatorics & Counting section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const combinatoricscountingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How many ways can you arrange 5 distinct books on a shelf?',
    options: ['25', '120', '625', '3,125'],
    correctAnswer: 1,
    explanation:
      'This is a permutation of all n items: n! = 5! = 5 × 4 × 3 × 2 × 1 = 120 ways. Order matters since different arrangements are different outcomes.',
  },
  {
    id: 'mc2',
    question:
      "How many ways can you select 3 features from 10 available features (order doesn't matter)?",
    options: ['30', '120', '720', '1,000'],
    correctAnswer: 1,
    explanation:
      "This is a combination: C(10,3) = 10!/(3!×7!) = (10×9×8)/(3×2×1) = 120. Order doesn't matter since selecting {A,B,C} is the same as {C,B,A}.",
  },
  {
    id: 'mc3',
    question: 'What is the difference between P(n,r) and C(n,r)?',
    options: [
      'P(n,r) considers order, C(n,r) does not',
      'C(n,r) considers order, P(n,r) does not',
      'They are the same thing with different notation',
      'P(n,r) is for small numbers, C(n,r) is for large numbers',
    ],
    correctAnswer: 0,
    explanation:
      "P(n,r) counts permutations where order matters, while C(n,r) counts combinations where order doesn't matter. In fact, C(n,r) = P(n,r)/r! because we divide out the r! orderings.",
  },
  {
    id: 'mc4',
    question:
      'If you have 4 hyperparameters, each with 5 possible values, how many total configurations are there in a grid search?',
    options: ['20', '120', '625', '1,024'],
    correctAnswer: 2,
    explanation:
      'Using the fundamental counting principle: 5 × 5 × 5 × 5 = 5^4 = 625 total configurations. Each parameter is chosen independently.',
  },
  {
    id: 'mc5',
    question:
      "What is the sum of all binomial coefficients in row n of Pascal's Triangle?",
    options: ['n', 'n!', '2^n', 'n^2'],
    correctAnswer: 2,
    explanation:
      'The sum C(n,0) + C(n,1) + ... + C(n,n) = 2^n. This equals the total number of subsets of an n-element set, since each element can either be included or excluded (2 choices per element).',
  },
];
