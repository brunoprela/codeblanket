/**
 * Multiple choice questions for The Algorithm Step-by-Step section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const algorithmMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'If the middle element is smaller than the target, what should you do next?',
    options: [
      'Set right = mid - 1',
      'Set left = mid + 1',
      'Set left = mid',
      'Return -1',
    ],
    correctAnswer: 1,
    explanation:
      'When the middle element is smaller than the target, the target must be in the right half (since the array is sorted). Update left = mid + 1 to search the right half, excluding the already-checked middle element.',
  },
  {
    id: 'mc2',
    question:
      'Why is mid = left + (right - left) // 2 preferred over mid = (left + right) // 2?',
    options: [
      'It is faster',
      'It prevents integer overflow in languages like Java/C++',
      'It gives a different result',
      'It uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      'The formula left + (right - left) // 2 prevents integer overflow that can occur when adding two large integers together. This is important in languages like Java and C++.',
  },
  {
    id: 'mc3',
    question:
      'Should the loop condition be "while left < right" or "while left <= right"?',
    options: [
      'while left < right',
      'while left <= right',
      'Both work the same',
      "It doesn't matter",
    ],
    correctAnswer: 1,
    explanation:
      'Use "while left <= right" to ensure the final element is checked when left equals right. Without the equal sign, you would skip checking the last element.',
  },
  {
    id: 'mc4',
    question: 'After finding that nums[mid] == target, what should you return?',
    options: ['target', 'mid', 'nums[mid]', 'true'],
    correctAnswer: 1,
    explanation:
      'Return mid, which is the index where the target was found. The problem typically asks for the index, not the value itself.',
  },
  {
    id: 'mc5',
    question:
      'If the loop exits without finding the target, what should you return?',
    options: ['0', '-1', 'null', 'false'],
    correctAnswer: 1,
    explanation:
      'Return -1 to indicate the target was not found in the array. This is the standard convention in most binary search implementations.',
  },
];
