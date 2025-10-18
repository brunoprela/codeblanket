/**
 * Multiple choice questions for Bitwise Operators section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const operatorsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does AND (&) operator do?',
    options: [
      'Adds bits',
      'Result 1 only if both bits are 1 - used for masking/clearing bits',
      'Flips bits',
      'Shifts bits',
    ],
    correctAnswer: 1,
    explanation:
      'AND (&): result bit is 1 only if both input bits are 1. Used for: masking (extract specific bits), clearing bits (AND with 0), checking if bit is set (x & (1<<i)).',
  },
  {
    id: 'mc2',
    question: 'What does XOR (^) operator do and what is its key property?',
    options: [
      'Multiplies',
      'Result 1 if bits differ - key property: x ^ x = 0, x ^ 0 = x (self-inverse)',
      'Adds',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'XOR (^): result 1 if bits differ. Key: self-inverse (x ^ x = 0, x ^ 0 = x). Used for: finding single number, swapping variables, detecting duplicates. Associative and commutative.',
  },
  {
    id: 'mc3',
    question: 'How do left shift (<<) and right shift (>>) work?',
    options: [
      'Add/subtract',
      'Left shift: multiply by 2^n (x << n = x * 2^n), Right: divide by 2^n',
      'Random',
      'Flip bits',
    ],
    correctAnswer: 1,
    explanation:
      'Left shift (x << n): shifts bits left, fills with 0, multiplies by 2^n. Right shift (x >> n): shifts right, divides by 2^n. Fast alternative to *2 or /2. Example: 5 << 1 = 10.',
  },
  {
    id: 'mc4',
    question: 'What does NOT (~) operator do?',
    options: [
      'Deletes bits',
      "Flips all bits: 0→1, 1→0 (one's complement)",
      'Adds 1',
      'Shifts',
    ],
    correctAnswer: 1,
    explanation:
      "NOT (~): flips every bit (one's complement). ~0 = all 1s, ~x produces negative number in two's complement. Used in formulas like ~x + 1 = -x.",
  },
  {
    id: 'mc5',
    question: 'How do you set/clear/toggle/check a specific bit?',
    options: [
      'Cannot do',
      'Set: x | (1<<i), Clear: x & ~(1<<i), Toggle: x ^ (1<<i), Check: x & (1<<i)',
      'Random',
      'Use arrays',
    ],
    correctAnswer: 1,
    explanation:
      'Bit operations: Set bit i: OR with 1<<i. Clear bit i: AND with ~(1<<i). Toggle bit i: XOR with 1<<i. Check bit i: AND with 1<<i (non-zero if set).',
  },
];
