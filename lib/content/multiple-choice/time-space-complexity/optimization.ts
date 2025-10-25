/**
 * Multiple choice questions for Optimization Strategies & Trade-offs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const optimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How can using a hash map improve an O(n²) nested loop algorithm?',
    options: [
      'By reducing space complexity',
      'By providing O(1) lookups to eliminate the inner loop',
      'By sorting the data faster',
      'By using less memory',
    ],
    correctAnswer: 1,
    explanation:
      'A hash map provides O(1) lookups, which can replace an inner O(n) loop with a single O(1) operation. For example, in two-sum, instead of checking every other element (O(n²)), we can check if the complement exists in the hash map (O(n)).',
  },
  {
    id: 'mc2',
    question: 'What is memoization and what complexity problem does it solve?',
    options: [
      'Sorting data to improve search time',
      'Caching function results to avoid redundant calculations',
      'Using less memory in recursive functions',
      'Converting recursive to iterative solutions',
    ],
    correctAnswer: 1,
    explanation:
      'Memoization caches the results of expensive function calls and returns the cached result when the same inputs occur again. It can dramatically reduce time complexity (e.g., Fibonacci from O(2ⁿ) to O(n)) at the cost of O(n) space.',
  },
  {
    id: 'mc3',
    question:
      'When would you prefer an O(n log n) algorithm over an O(n²) algorithm?',
    options: [
      'Never, O(n²) is always faster',
      'For very small inputs only',
      'For large inputs where better scaling matters',
      'When you need to use less memory',
    ],
    correctAnswer: 2,
    explanation:
      'For large inputs, O(n log n) scales much better than O(n²). While O(n²) might be faster for very small inputs due to lower constants, O(n log n) becomes dramatically faster as n grows (e.g., n=1000: n log n ≈ 10,000 vs n² = 1,000,000).',
  },
  {
    id: 'mc4',
    question:
      'What optimization technique does this demonstrate?\n\n```python\nprefix_sum = [0]\nfor num in arr:\n    prefix_sum.append (prefix_sum[-1] + num)\n```',
    options: [
      'Memoization',
      'Two pointers',
      'Precomputation/preprocessing',
      'Binary search',
    ],
    correctAnswer: 2,
    explanation:
      'This is precomputation - calculating prefix sums upfront (O(n)) to enable O(1) range sum queries later. This trades O(n) space and preprocessing time for much faster subsequent queries.',
  },
  {
    id: 'mc5',
    question: 'Which is a valid time-space tradeoff?',
    options: [
      'Using less memory always makes algorithms faster',
      'Using more memory can sometimes make algorithms faster',
      'Time and space complexity must always be equal',
      'Optimization always improves both time and space',
    ],
    correctAnswer: 1,
    explanation:
      'Time-space tradeoffs often involve using more memory to achieve better time complexity. Examples include hash maps for O(1) lookups, memoization for avoiding recalculation, and prefix sums for fast range queries.',
  },
];
