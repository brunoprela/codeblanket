/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of a fixed-size sliding window algorithm?',
    options: ['O(k)', 'O(n)', 'O(n*k)', 'O(n²)'],
    correctAnswer: 1,
    explanation:
      'Fixed-size sliding window is O(n) because after computing the initial window in O(k), each of the remaining n-k windows is updated in O(1) time, giving O(k + n - k) = O(n).',
  },
  {
    id: 'mc2',
    question:
      'Why is variable-size sliding window O(n) despite having a nested while loop?',
    options: [
      'The while loop never executes',
      'The left pointer moves at most n times total across all iterations',
      'It uses memoization',
      'It is actually O(n²)',
    ],
    correctAnswer: 1,
    explanation:
      "Although there's a nested while loop, the left pointer can only move from 0 to n-1 throughout the entire algorithm. So the inner loop executes O(n) times total, not per outer iteration.",
  },
  {
    id: 'mc3',
    question:
      'What is the space complexity when using a hash set to track characters in a window?',
    options: [
      'O(1)',
      'O(k) where k is window size or character set size',
      'O(n)',
      'O(n²)',
    ],
    correctAnswer: 1,
    explanation:
      "Space is O(k) where k is the size of the window or character set. For lowercase English letters, k ≤ 26, so it's O(26) = O(1) constant space.",
  },
  {
    id: 'mc4',
    question:
      'How does sliding window improve upon brute force for maximum sum of k elements?',
    options: [
      'From O(n²) to O(n)',
      'From O(n*k) to O(n)',
      'From O(n) to O(log n)',
      'No improvement',
    ],
    correctAnswer: 1,
    explanation:
      'Brute force recalculates each k-element window sum in O(k) time for O(n) windows, giving O(n*k). Sliding window reuses the previous sum and updates in O(1), achieving O(n).',
  },
  {
    id: 'mc5',
    question:
      'For longest substring without repeating characters, what is the brute force complexity vs sliding window?',
    options: [
      'O(n²) vs O(n)',
      'O(n³) vs O(n)',
      'O(n) vs O(log n)',
      'Same complexity',
    ],
    correctAnswer: 1,
    explanation:
      'Brute force checks all O(n²) substrings and verifies each in O(n) time for duplicates, giving O(n³). Sliding window with hash set does a single pass in O(n).',
  },
];
