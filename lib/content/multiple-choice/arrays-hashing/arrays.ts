/**
 * Multiple choice questions for Array Fundamentals & Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const arraysMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of accessing an element by index in an array?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 0,
    explanation:
      'Array access by index is O(1) constant time because the memory address can be calculated directly using: base_address + (index × element_size).',
  },
  {
    id: 'mc2',
    question: "What does Kadane\'s algorithm solve?",
    options: [
      'Maximum element in array',
      'Maximum subarray sum',
      'Minimum subarray sum',
      'Array sorting',
    ],
    correctAnswer: 1,
    explanation:
      "Kadane\'s algorithm finds the maximum sum of any contiguous subarray in O(n) time by maintaining the maximum sum ending at each position.",
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of building a prefix sum array?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'Building a prefix sum array requires iterating through the array once to compute cumulative sums, which takes O(n) time. Once built, range queries can be answered in O(1).',
  },
  {
    id: 'mc4',
    question:
      'Why is inserting an element at the beginning of an array O(n) instead of O(1)?',
    options: [
      'The array must be sorted',
      'All existing elements must be shifted to the right',
      'The array must be resized',
      'The element must be hashed first',
    ],
    correctAnswer: 1,
    explanation:
      'Inserting at the beginning requires shifting all n existing elements one position to the right to make room, resulting in O(n) time complexity.',
  },
  {
    id: 'mc5',
    question:
      'In the sliding window technique, how is the window sum updated when moving to the next position?',
    options: [
      'Recalculate the entire sum',
      'Add the new element and subtract the old element',
      'Use binary search',
      'Use a hash table',
    ],
    correctAnswer: 1,
    explanation:
      'The sliding window technique updates the sum in O(1) by adding the element entering the window and subtracting the element leaving the window, avoiding recalculation.',
  },
];
