/**
 * Multiple choice questions for Advanced Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What data structure is used for sliding window maximum with O(n) time?',
    options: ['Stack', 'Monotonic deque', 'Hash map', 'Binary tree'],
    correctAnswer: 1,
    explanation:
      'A monotonic deque maintains elements in decreasing order of values, allowing O(1) access to the maximum. Each element is added and removed at most once, achieving O(n) total time.',
  },
  {
    id: 'mc2',
    question:
      'For "at most k distinct characters", what auxiliary structure do you typically use?',
    options: [
      'Array',
      'Hash map to count character frequencies',
      'Binary search tree',
      'Linked list',
    ],
    correctAnswer: 1,
    explanation:
      'A hash map tracks character frequencies in the current window. When the map size exceeds k, you have too many distinct characters and need to shrink the window.',
  },
  {
    id: 'mc3',
    question:
      'What is the prefix sum technique and how does it relate to sliding windows?',
    options: [
      'They are unrelated',
      'Prefix sum can solve subarray sum problems that sliding window cannot handle (negative numbers)',
      'Prefix sum is slower than sliding window',
      'They always give the same solution',
    ],
    correctAnswer: 1,
    explanation:
      'Prefix sum with hash map handles subarray sum problems with negative numbers, which pure sliding window cannot. Sliding window requires monotonic behavior (window sum increases/decreases predictably).',
  },
  {
    id: 'mc4',
    question:
      'In anagram detection problems, how do you verify if two character frequency maps are equal?',
    options: [
      'Compare each character count',
      'Use Counter equality (window_count == p_count)',
      'Manually iterate',
      'Sort both strings',
    ],
    correctAnswer: 1,
    explanation:
      "Python's Counter objects can be directly compared for equality, checking if all character frequencies match. This is cleaner and more efficient than manual iteration.",
  },
  {
    id: 'mc5',
    question:
      'What makes a problem suitable for sliding window vs other techniques?',
    options: [
      'Any array problem',
      'Problems involving contiguous sequences with local properties',
      'Only sorted array problems',
      'Problems requiring global information',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding window works for contiguous sequences where the window state can be incrementally updated. Problems requiring non-local or global information typically need other techniques.',
  },
];
