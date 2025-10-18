/**
 * Multiple choice questions for Advanced Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedtechniquesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a bitmask and how is it used?',
    options: [
      'Random mask',
      'Integer where each bit represents element in set - enables O(1) set operations',
      'String',
      'Array',
    ],
    correctAnswer: 1,
    explanation:
      'Bitmask: use integer bits to represent set. Bit i set = element i in set. Operations: union (OR), intersection (AND), add (OR with 1<<i), remove (AND with ~(1<<i)). O(1) operations.',
  },
  {
    id: 'mc2',
    question: 'How do you generate all subsets using bit manipulation?',
    options: [
      'Cannot do',
      "Iterate 0 to 2^n-1, each number's bits indicate which elements to include",
      'Backtracking only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'For n elements, iterate i from 0 to 2^n-1. Each i represents subset: if bit j set in i, include element j. Example: for [a,b,c], i=5 (101) = [a,c]. O(2^n) time.',
  },
  {
    id: 'mc3',
    question: 'What is Gray code?',
    options: [
      'Random code',
      'Binary code where adjacent numbers differ by exactly one bit',
      'Compression',
      'Error code',
    ],
    correctAnswer: 1,
    explanation:
      'Gray code: binary sequence where consecutive numbers differ by one bit. Formula: G(n) = n ^ (n >> 1). Used in hardware, rotary encoders to avoid spurious transitions. Example: 0→1→3→2 (00→01→11→10).',
  },
  {
    id: 'mc4',
    question: 'How do you find the k-th bit in a number?',
    options: [
      'Cannot do',
      '(n >> k) & 1 - right shift k positions, check LSB',
      'Division',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Get k-th bit: (n >> k) & 1. Right shift moves k-th bit to position 0, AND with 1 extracts it. Or use n & (1 << k) and check if non-zero. Both O(1).',
  },
  {
    id: 'mc5',
    question: 'What is bit packing?',
    options: [
      'Random technique',
      'Store multiple values in single integer using different bit ranges - save space',
      'Compression',
      'Encryption',
    ],
    correctAnswer: 1,
    explanation:
      'Bit packing: store multiple small values in one integer. RGB color: 8 bits red, 8 green, 8 blue in 24-bit int. Extract: (color >> 16) & 0xFF for red. Saves space vs separate variables.',
  },
];
