/**
 * Multiple choice questions for Combinatorics and Sequences section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const combinatoricsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the difference between permutations and combinations?',
    options: [
      'Same',
      "Permutations: order matters (nPr = n!/(n-r)!). Combinations: order doesn't matter (nCr = n!/(r!(n-r)!))",
      'Random',
      'Opposite',
    ],
    correctAnswer: 1,
    explanation:
      "Permutations: arrangements where order matters. P(n,r) = n!/(n-r)!. Example: ABC vs ACB are different. Combinations: selections where order doesn't matter. C(n,r) = n!/(r!(n-r)!). ABC = ACB.",
  },
  {
    id: 'mc2',
    question: 'How do you compute large factorials modulo m?',
    options: [
      'Compute then mod',
      'Multiply and apply mod at each step: fact = (fact * i) % m to prevent overflow',
      'Cannot compute',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Large factorial: compute iteratively, apply mod after each multiplication to prevent overflow. fact = 1; for i in 2..n: fact = (fact * i) % m. Property: (a*b)%m = ((a%m)*(b%m))%m.',
  },
  {
    id: 'mc3',
    question: "What is Pascal\'s Triangle and its use?",
    options: [
      'Random triangle',
      'Triangle where C(n,r) = C(n-1,r-1) + C(n-1,r) - computes combinations efficiently',
      'Geometry',
      'No use',
    ],
    correctAnswer: 1,
    explanation:
      "Pascal\'s Triangle: each entry is sum of two above. Row n contains C(n,0), C(n,1),...,C(n,n). Property: C(n,r) = C(n-1,r-1) + C(n-1,r). Precompute combinations in O(n²).",
  },
  {
    id: 'mc4',
    question:
      'How do you count number of ways to climb n stairs (1 or 2 steps)?',
    options: [
      'n',
      'Fibonacci: ways (n) = ways (n-1) + ways (n-2) - sum of ways from 1-step and 2-step before',
      '2^n',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Climbing stairs: can reach step n from n-1 (1 step) or n-2 (2 steps). ways (n) = ways (n-1) + ways (n-2). Same as Fibonacci. Base: ways(1)=1, ways(2)=2.',
  },
  {
    id: 'mc5',
    question: 'What is the Pigeonhole Principle?',
    options: [
      'Random principle',
      'If n+1 items in n boxes, at least one box has 2+ items - guarantees collision',
      'Sorting',
      'No principle',
    ],
    correctAnswer: 1,
    explanation:
      'Pigeonhole: if more items than boxes, at least one box contains multiple items. Used in proofs: n+1 birthdays in n days → 2 people share birthday. Guarantees duplicates/collisions.',
  },
];
