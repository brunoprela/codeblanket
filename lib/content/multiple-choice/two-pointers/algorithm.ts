/**
 * Multiple choice questions for Detailed Algorithm Walkthrough section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const algorithmMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of the two sum algorithm using two pointers on a sorted array?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'The two pointers algorithm runs in O(n) time because each pointer traverses the array at most once, and combined they cover all elements in a single pass.',
  },
  {
    id: 'mc2',
    question:
      'In the two sum algorithm, when the current sum equals the target, what should you do?',
    options: [
      'Continue searching for more pairs',
      'Return the indices immediately',
      'Move both pointers',
      'Sort the array again',
    ],
    correctAnswer: 1,
    explanation:
      "When you find a pair whose sum equals the target, you can return the indices immediately since you've found the solution.",
  },
  {
    id: 'mc3',
    question:
      'In the remove duplicates algorithm, what is the initial value of the slow pointer?',
    options: ['-1', '0', '1', 'len(array) - 1'],
    correctAnswer: 1,
    explanation:
      'The slow pointer starts at 0 (the first element) since the first element is always unique. The fast pointer typically starts at 1 to begin comparing.',
  },
  {
    id: 'mc4',
    question: 'What is the return value of the remove duplicates function?',
    options: [
      'The modified array',
      'The number of unique elements',
      'A list of duplicates',
      'True or False',
    ],
    correctAnswer: 1,
    explanation:
      'The function returns slow + 1, which represents the length/count of unique elements. The array is modified in-place, and the first slow + 1 elements contain the unique values.',
  },
  {
    id: 'mc5',
    question:
      'Why is sorting a prerequisite for the two sum two-pointers algorithm?',
    options: [
      'To make the code simpler',
      'To provide predictable behavior when moving pointers',
      'To reduce space complexity',
      'It is not actually required',
    ],
    correctAnswer: 1,
    explanation:
      'Sorting ensures that moving a pointer in one direction consistently increases or decreases values, allowing reliable decisions about which pointer to move based on whether the sum is too large or too small.',
  },
];
