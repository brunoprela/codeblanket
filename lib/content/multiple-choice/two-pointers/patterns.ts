/**
 * Multiple choice questions for The Three Main Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In the opposite direction pattern, when do the pointers stop moving?',
    options: [
      'When they point to the same element or cross',
      'When they reach the middle of the array',
      'After n iterations',
      'When one pointer reaches the end',
    ],
    correctAnswer: 0,
    explanation:
      'In the opposite direction pattern, pointers continue until they meet (point to the same element) or cross each other (left > right). This ensures all elements are considered.',
  },
  {
    id: 'mc2',
    question:
      'What is the key characteristic of the fast and slow pointer pattern?',
    options: [
      'Both pointers move at the same speed',
      'Pointers move from opposite ends',
      'Pointers start at the same position but move at different speeds',
      'One pointer is always twice as fast as the other',
    ],
    correctAnswer: 2,
    explanation:
      'The fast and slow pointer pattern has both pointers starting at the beginning (or same position) but moving at different speeds - often the fast pointer moves every iteration while the slow only moves conditionally.',
  },
  {
    id: 'mc3',
    question:
      'For the two sum problem on a sorted array, if the current sum is too large, which pointer should you move?',
    options: [
      'Move the left pointer right',
      'Move the right pointer left',
      'Move both pointers',
      "It doesn't matter",
    ],
    correctAnswer: 1,
    explanation:
      'If the sum is too large, you need smaller numbers. Since the array is sorted, moving the right pointer left gives you a smaller value, reducing the sum.',
  },
  {
    id: 'mc4',
    question:
      'What problem type is the sliding window pattern best suited for?',
    options: [
      'Finding pairs in sorted arrays',
      'Checking palindromes',
      'Subarray problems with size or property constraints',
      'Removing duplicates',
    ],
    correctAnswer: 2,
    explanation:
      'Sliding window is ideal for subarray problems where you need to maintain a window with certain constraints (fixed size, max sum, unique characters, etc.), expanding or shrinking as needed.',
  },
  {
    id: 'mc5',
    question:
      'In the remove duplicates problem, what does the slow pointer represent?',
    options: [
      'The current element being checked',
      'The position where the next unique element should be written',
      'The last duplicate found',
      'The middle of the array',
    ],
    correctAnswer: 1,
    explanation:
      'The slow pointer marks the end of the unique elements section - it points to the last unique element, so the next unique element will be written at slow + 1.',
  },
];
