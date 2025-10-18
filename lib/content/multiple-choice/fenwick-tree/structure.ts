/**
 * Multiple choice questions for Fenwick Tree Structure section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const structureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does i & -i compute in Fenwick Tree?',
    options: [
      'Random value',
      'Isolates the last set bit (rightmost 1) - determines range size',
      'Doubles i',
      'Negates i',
    ],
    correctAnswer: 1,
    explanation:
      "i & -i isolates the last set bit using two's complement trick. This determines how many elements index i is responsible for. Example: 12 (1100) & -12 = 4 (0100), so index 12 covers 4 elements.",
  },
  {
    id: 'mc2',
    question: 'How do you find the parent of index i in Fenwick Tree?',
    options: ['i/2', 'i + (i & -i) - add last set bit', 'i-1', 'i*2'],
    correctAnswer: 1,
    explanation:
      'Parent of i is i + (i & -i). This adds the last set bit, moving to the next index responsible for a larger range. Example: parent of 4 (100) is 4+4=8 (1000).',
  },
  {
    id: 'mc3',
    question: 'What range does index i cover in Fenwick Tree?',
    options: [
      'Single element',
      'Range [i - (i & -i) + 1, i] with length (i & -i)',
      'Entire array',
      'Random range',
    ],
    correctAnswer: 1,
    explanation:
      'Index i covers (i & -i) elements ending at position i. Range is [i - (i & -i) + 1, i]. Example: index 6 (110) covers 6 & -6 = 2 elements: positions 5 and 6.',
  },
  {
    id: 'mc4',
    question: 'Why are Fenwick Tree indices 1-based?',
    options: [
      'Random choice',
      'Bit manipulation (i & -i) fails for index 0 - would be 0',
      'Easier to read',
      'Historical reasons',
    ],
    correctAnswer: 1,
    explanation:
      'Index 0 has no set bits, so 0 & -0 = 0, breaking the algorithm. Starting at index 1 ensures all indices have at least one set bit, making bit manipulation work correctly.',
  },
  {
    id: 'mc5',
    question: 'What makes Fenwick Tree elegant compared to Segment Tree?',
    options: [
      'Faster',
      'Uses bit manipulation tricks for parent/child navigation - no explicit tree structure',
      'More powerful',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree uses bit manipulation (i & -i, i + (i & -i), i - (i & -i)) to implicitly represent tree structure in a simple array. No pointers or complex indexing like Segment Tree.',
  },
];
