/**
 * Multiple choice questions for Advanced Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a 2D Fenwick Tree used for?',
    options: [
      'Sorting 2D arrays',
      'Range sum queries on 2D matrix (rectangle sums) in O(log²N) time',
      'Graph traversal',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      '2D Fenwick Tree handles 2D range queries like rectangle sums. Uses nested Fenwick Trees: outer for rows, inner for columns. Update and query both O(log M × log N).',
  },
  {
    id: 'mc2',
    question: 'How do you count inversions using Fenwick Tree?',
    options: [
      'Cannot do',
      'For each element, count how many larger elements came before (coordinate compression + range query)',
      'Sort array',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Inversions: process left to right. For each element x, query sum of elements > x seen so far. Update tree at position x. With coordinate compression for space efficiency. O(N log N) time.',
  },
  {
    id: 'mc3',
    question: 'What does coordinate compression do for Fenwick Tree?',
    options: [
      'Compresses data',
      'Maps large value range to smaller [1, N] for space efficiency',
      'Random',
      'Deletes data',
    ],
    correctAnswer: 1,
    explanation:
      'When values are large (e.g., 10^9) but only N distinct values, map them to [1,N]. Sorts unique values, creates mapping. Fenwick Tree size becomes O(N) instead of O(max_value).',
  },
  {
    id: 'mc4',
    question: 'Can Fenwick Tree handle range updates efficiently?',
    options: [
      'No',
      'Yes with difference array technique - update endpoints O(log N)',
      'Only point updates',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Range update [L,R] += delta: use difference array. Update diff[L] += delta, diff[R+1] -= delta. Query reconstructs by prefix sum of differences. Range update becomes O(log N).',
  },
  {
    id: 'mc5',
    question: 'What makes Fenwick Tree implementation elegant?',
    options: [
      'Random',
      'Just 2 simple functions using bit manipulation - ~20 lines total',
      'Uses complex data structures',
      'Always fast',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree beauty: entire implementation is 2 functions (update, prefix_sum), each 3-5 lines with simple bit manipulation. No complex tree structures or recursion. Very interview-friendly.',
  },
];
