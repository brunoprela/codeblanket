/**
 * Multiple choice questions for Advanced Variations & Applications section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const variationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a monotonic function?',
    options: [
      'A function that is random',
      'A function that is always increasing or always decreasing',
      'A function that has multiple peaks',
      'A function that is constant',
    ],
    correctAnswer: 1,
    explanation:
      'A monotonic function is one that consistently increases or consistently decreases, never changing direction. This property is crucial for binary search because it allows reliable decision-making.',
  },
  {
    id: 'mc2',
    question:
      'In a rotated sorted array like [4,5,6,7,0,1,2], what is the key insight for using binary search?',
    options: [
      'Sort it first',
      'At least one half is always properly sorted',
      'Use linear search instead',
      'Find the rotation point first',
    ],
    correctAnswer: 1,
    explanation:
      'Even though the array is rotated, at least one half (left or right of mid) is always properly sorted. You can determine which half is sorted and decide where to search.',
  },
  {
    id: 'mc3',
    question:
      'What keywords in a problem statement suggest binary search might apply?',
    options: [
      'Sum, average, total',
      'First, last, minimum, maximum, at least',
      'Count, frequency, duplicate',
      'Random, shuffle, permutation',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "first", "last", "minimum", "maximum", or "at least/most" often indicate optimization or threshold problems that can be solved with binary search on a monotonic property.',
  },
  {
    id: 'mc4',
    question:
      'How can you use binary search to find square root of 25 without using sqrt()?',
    options: [
      'Binary search from 1 to 25, checking if mid * mid equals, is less than, or greater than 25',
      'Use linear search',
      'Divide 25 by 2 repeatedly',
      'Cannot be done with binary search',
    ],
    correctAnswer: 0,
    explanation:
      'Binary search on the range [0, 25]. The function f (x) = x² is monotonic increasing, so you can check if mid² is too small (search right), too big (search left), or equal (found).',
  },
  {
    id: 'mc5',
    question:
      'What is the key question to ask yourself to recognize if binary search applies?',
    options: [
      'Is the array large?',
      'If I check a value, can I tell whether I need to go higher or lower?',
      'Does the array have duplicates?',
      'Is the problem about trees?',
    ],
    correctAnswer: 1,
    explanation:
      'The hallmark of binary search problems is the ability to check a value and determine direction (higher/lower, left/right). This indicates a monotonic property you can exploit.',
  },
];
