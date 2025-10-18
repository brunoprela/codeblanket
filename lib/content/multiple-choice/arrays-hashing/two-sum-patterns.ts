/**
 * Multiple choice questions for Two-Sum Patterns Family section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const twosumpatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of Two Sum using a hash table?',
    options: ['O(n²)', 'O(n log n)', 'O(n)', 'O(1)'],
    correctAnswer: 2,
    explanation:
      'Two Sum with hash table is O(n) time - single pass through array with O(1) hash lookups for each element. We check and insert each element once.',
  },
  {
    id: 'mc2',
    question:
      'For Two Sum II (sorted array), what is the space complexity using two pointers?',
    options: ['O(n)', 'O(log n)', 'O(1)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'Two pointers approach uses O(1) space - only two pointer variables. We leverage the sorted property instead of extra data structures.',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of 3Sum using sort + two pointers?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
    correctAnswer: 2,
    explanation:
      '3Sum is O(n²): O(n log n) for sorting + O(n²) for fix-one-element (n times) × two-pointer-search (n). The n² dominates, so overall O(n²).',
  },
  {
    id: 'mc4',
    question: 'Why must we skip duplicates in 3Sum?',
    options: [
      'To improve performance',
      'To ensure unique triplets in the result',
      'To reduce space usage',
      'To handle negative numbers',
    ],
    correctAnswer: 1,
    explanation:
      'We skip duplicates to ensure unique triplets. Without skipping, we would find the same triplet multiple times. For example, [-1,-1,2] might be found twice if we do not skip the second -1.',
  },
  {
    id: 'mc5',
    question: 'For 4Sum, what is the time complexity?',
    options: ['O(n²)', 'O(n³)', 'O(n⁴)', 'O(n log n)'],
    correctAnswer: 1,
    explanation:
      '4Sum using sort + nested loops + two pointers is O(n³): fix two elements (n²) × two-pointer search (n) = O(n³).',
  },
];
