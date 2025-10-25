/**
 * Multiple choice questions for Advanced Techniques & Variations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In the Container With Most Water problem, which pointer should you move?',
    options: [
      'Always move the left pointer',
      'Always move the right pointer',
      'Move the pointer at the shorter line',
      'Move the pointer at the taller line',
    ],
    correctAnswer: 2,
    explanation:
      'Always move the pointer at the shorter line because the area is limited by the shorter height. Moving the taller pointer can only decrease or maintain the area, while moving the shorter pointer gives a chance to find a taller line.',
  },
  {
    id: 'mc2',
    question:
      'What is the time complexity of the 3Sum problem using two pointers?',
    options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
    correctAnswer: 2,
    explanation:
      "3Sum uses an outer loop O(n) with two pointers inside O(n), resulting in O(n²) time. Although sorting takes O(n log n), it's dominated by the O(n²) nested operations.",
  },
  {
    id: 'mc3',
    question:
      "In Floyd\'s cycle detection, what is the speed ratio between fast and slow pointers?",
    options: [
      'Fast is 3x slow',
      'Fast is 2x slow',
      'They move at same speed',
      'Fast is 4x slow',
    ],
    correctAnswer: 1,
    explanation:
      "In Floyd\'s cycle detection algorithm, the slow pointer moves 1 step at a time while the fast pointer moves 2 steps at a time - a 2:1 speed ratio.",
  },
  {
    id: 'mc4',
    question:
      'For the Dutch National Flag problem (sort 0s, 1s, 2s), how many pointers are typically used?',
    options: ['One', 'Two', 'Three', 'Four'],
    correctAnswer: 2,
    explanation:
      'The Dutch National Flag problem uses three pointers: one for the boundary of 0s, one for the boundary of 2s, and one for the current element being examined.',
  },
  {
    id: 'mc5',
    question: 'In 3Sum, why must you skip duplicate values?',
    options: [
      'To improve time complexity',
      'To avoid returning duplicate triplets in the result',
      'To reduce space complexity',
      'Because the algorithm breaks otherwise',
    ],
    correctAnswer: 1,
    explanation:
      'Skipping duplicates prevents returning multiple instances of the same triplet. For example, if the array has [1,1,1], you want to return each unique combination only once.',
  },
];
