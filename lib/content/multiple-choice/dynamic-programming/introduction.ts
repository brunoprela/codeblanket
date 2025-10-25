/**
 * Multiple choice questions for Introduction to Dynamic Programming section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What are the two key properties for Dynamic Programming?',
    options: [
      'Fast and simple',
      'Optimal substructure and overlapping subproblems',
      'Recursion and iteration',
      'Space and time',
    ],
    correctAnswer: 1,
    explanation:
      'DP requires: 1) Optimal substructure - optimal solution built from optimal subproblems, 2) Overlapping subproblems - same subproblems solved multiple times. Both must be present.',
  },
  {
    id: 'mc2',
    question: 'What is the difference between top-down and bottom-up DP?',
    options: [
      'No difference',
      'Top-down: recursive with memoization (cache). Bottom-up: iterative with tabulation (table)',
      'Top-down is always better',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Top-down (memoization): start with original problem, recurse, cache results. Bottom-up (tabulation): start with base cases, iteratively build up. Top-down more intuitive, bottom-up more space-efficient.',
  },
  {
    id: 'mc3',
    question:
      'Why does DP improve time complexity for Fibonacci from O(2^N) to O(N)?',
    options: [
      'Different algorithm',
      'Caches subproblem results - each fib (i) computed once instead of exponentially many times',
      'Uses more space',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Naive recursion recomputes fib(2) many times = O(2^N). DP caches each fib (i) result. Each of N subproblems computed once = O(N). Trades space for time.',
  },
  {
    id: 'mc4',
    question: 'When should you use DP instead of greedy?',
    options: [
      'Always',
      "When greedy doesn't give optimal (e.g., coin change with arbitrary denominations)",
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use DP when greedy fails to give optimal solution. Example: coin change [1,3,4] for amount 6. Greedy: 4+1+1=3 coins. DP optimal: 3+3=2 coins. DP tries all possibilities.',
  },
  {
    id: 'mc5',
    question: 'What is memoization?',
    options: [
      'Memory management',
      'Caching results of expensive function calls to avoid recomputation',
      'Writing notes',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Memoization: cache function results in dictionary/map. When function called with same args, return cached result instead of recomputing. Top-down DP approach. Trade-off: space for time.',
  },
];
