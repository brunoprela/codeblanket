/**
 * Multiple choice questions for Common Bit Manipulation Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonpatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do you check if a number is a power of 2?',
    options: [
      'Try dividing',
      'n & (n-1) == 0 and n != 0 - power of 2 has single set bit',
      'Loop',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Power of 2 has exactly one set bit: 8 = 1000. Formula: n & (n-1) clears rightmost set bit. If result is 0 and n>0, only one bit was set = power of 2.',
  },
  {
    id: 'mc2',
    question: "What is Brian Kernighan's algorithm?",
    options: [
      'Sorting',
      'Count set bits by repeatedly clearing rightmost set bit: n = n & (n-1) until n=0',
      'Search',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Brian Kernighan's algorithm counts set bits efficiently. n & (n-1) clears rightmost set bit. Repeat until n=0, counting iterations. O(number of set bits) instead of O(log N).",
  },
  {
    id: 'mc3',
    question: 'How do you isolate the rightmost set bit?',
    options: [
      'Cannot do',
      'n & -n or n & (~n + 1) - gives lowest set bit only',
      'n & 1',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Isolate rightmost set bit: n & -n. Works because -n is two's complement (~n + 1). All bits after rightmost 1 flip, and AND keeps only that bit. Example: 12 (1100) & -12 = 4 (0100).",
  },
  {
    id: 'mc4',
    question: 'How do you swap two numbers without a temporary variable?',
    options: [
      'Impossible',
      'XOR swap: a ^= b, b ^= a, a ^= b - uses XOR self-inverse property',
      'Addition',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'XOR swap: a ^= b makes a = a^b, b ^= a makes b = b^(a^b) = original a, a ^= b makes a = (a^b)^b = original b. Works because XOR is self-inverse. Caveat: a and b must be different addresses.',
  },
  {
    id: 'mc5',
    question: 'How do you check if a number is even?',
    options: [
      'n % 2',
      'n & 1 == 0 - check least significant bit (LSB)',
      'Division',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Check even: n & 1 == 0. LSB is 0 for even, 1 for odd. Faster than modulo. Similarly, check odd: n & 1 == 1.',
  },
];
