/**
 * Multiple choice questions for Memoization: Optimizing Recursion section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const memoizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-memo1',
    question: 'What is the time complexity of memoized Fibonacci?',
    options: ['O(2^n)', 'O(n)', 'O(n log n)', 'O(n²)'],
    correctAnswer: 1,
    explanation:
      "With memoization, each Fibonacci number is computed exactly once, so n computations total = O(n). Without memoization it's O(2^n).",
  },
  {
    id: 'mc-memo2',
    question: 'What is the space complexity of memoized recursion?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'Same as before memoization'],
    correctAnswer: 2,
    explanation:
      'Memoization requires O(n) space to store cached results for n unique inputs, plus O(n) for the recursion call stack.',
  },
  {
    id: 'mc-memo3',
    question: 'Which Python decorator is used for automatic memoization?',
    options: ['@cache', '@lru_cache', '@memoize', '@remember'],
    correctAnswer: 1,
    explanation:
      '@lru_cache from functools module provides automatic memoization with LRU (Least Recently Used) cache eviction policy.',
  },
  {
    id: 'mc-memo4',
    question: 'What type of problems benefit most from memoization?',
    options: [
      'Problems with no repeated subproblems',
      'Problems with overlapping subproblems',
      'Problems with simple linear recursion',
      'Problems with external state dependencies',
    ],
    correctAnswer: 1,
    explanation:
      'Memoization shines with overlapping subproblems where the same computation is repeated multiple times (e.g., Fibonacci, DP problems).',
  },
  {
    id: 'mc-memo5',
    question: 'What is the main trade-off of using memoization?',
    options: [
      'Slower execution for faster space',
      'More complex code for no benefit',
      'Memory usage for time savings',
      'Less accurate results for speed',
    ],
    correctAnswer: 2,
    explanation:
      'Memoization trades increased memory usage (storing cached results) for dramatically reduced time complexity by avoiding redundant computations.',
  },
];
