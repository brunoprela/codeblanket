/**
 * Multiple choice questions for Code Templates & Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In the "pair with sum" template, what do you return when no pair is found?',
    options: [
      'null',
      'Empty array or [-1, -1]',
      'The closest pair',
      'An error',
    ],
    correctAnswer: 1,
    explanation:
      'When no pair sums to the target, the function typically returns an empty array [] or [-1, -1] to indicate no solution was found.',
  },
  {
    id: 'mc2',
    question:
      'In the "remove element" template, what does the slow pointer represent?',
    options: [
      'Elements to be removed',
      'The next position to write a kept element',
      'The current element being checked',
      'The end of the array',
    ],
    correctAnswer: 1,
    explanation:
      'The slow pointer marks the position where the next element that should be kept (not removed) will be written.',
  },
  {
    id: 'mc3',
    question: 'For the 3Sum problem, what is the overall time complexity?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
    correctAnswer: 2,
    explanation:
      '3Sum requires sorting (O(n log n)) plus an outer loop (O(n)) with two pointers inside (O(n)), giving O(n) × O(n) = O(n²). The sorting is dominated by the nested operations.',
  },
  {
    id: 'mc4',
    question: 'In the partition template, when do you swap elements?',
    options: [
      'After every iteration',
      'When left finds an element for right section AND right finds one for left section',
      'Only at the end',
      'When pointers meet',
    ],
    correctAnswer: 1,
    explanation:
      'You swap when the left pointer finds an element that belongs in the right section AND the right pointer finds an element that belongs in the left section - both conditions must be true.',
  },
  {
    id: 'mc5',
    question: 'What pattern would you use for the "move zeros to end" problem?',
    options: [
      'Opposite direction',
      'Sliding window',
      'Same direction (fast & slow)',
      'Binary search',
    ],
    correctAnswer: 2,
    explanation:
      'Move zeros uses the same direction pattern where slow marks position for next non-zero, and fast scans ahead to find non-zero elements to move.',
  },
];
