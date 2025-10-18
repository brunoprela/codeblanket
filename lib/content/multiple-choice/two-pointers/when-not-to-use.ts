/**
 * Multiple choice questions for When NOT to Use Two Pointers section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const whennottouseMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'When does two pointers fail for pair-finding problems?',
    options: [
      'When the array is sorted',
      'When you need the original indices',
      'When there is only one pair',
      'When the array is small',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers usually requires sorting, which loses original indices. If the problem asks for original indices (like Two Sum), use a hash map instead.',
  },
  {
    id: 'mc2',
    question: 'Why is two pointers NOT suitable for finding all permutations?',
    options: [
      'Too slow',
      'Requires too much space',
      'Cannot backtrack to explore different orderings',
      'Only works on sorted arrays',
    ],
    correctAnswer: 2,
    explanation:
      'Two pointers moves forward/backward in a single pass. Permutations require backtracking - making choices, exploring, and undoing to try alternatives.',
  },
  {
    id: 'mc3',
    question:
      'What should you use instead of two pointers for "find all pairs with sum = target"?',
    options: [
      'Binary search',
      'Hash map or nested loops',
      'Dynamic programming',
      'Divide and conquer',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers finds ONE pair efficiently. For ALL pairs (especially with duplicates), use hash map to count occurrences or nested loops if small.',
  },
  {
    id: 'mc4',
    question: 'When is two pointers inappropriate for 2D matrix problems?',
    options: [
      'When matrix is sorted',
      'When you need to explore regions or traverse in multiple directions',
      'When matrix is small',
      'Never, two pointers always works',
    ],
    correctAnswer: 1,
    explanation:
      'Two pointers is fundamentally 1D. For 2D region exploration, connected components, or multi-directional traversal, use DFS/BFS or DP.',
  },
  {
    id: 'mc5',
    question:
      'You need maximum sum of TWO non-overlapping subarrays. Why not use sliding window?',
    options: [
      'Sliding window is too slow',
      'Sliding window tracks one window, cannot track two independent windows simultaneously',
      'Sliding window only works on sorted arrays',
      'Sliding window uses too much space',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding window (two-pointer technique) maintains one contiguous window. For multiple independent windows, use DP or prefix sums to track best previous windows.',
  },
];
