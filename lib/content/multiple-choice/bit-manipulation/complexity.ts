/**
 * Multiple choice questions for Time and Space Complexity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of basic bitwise operations?',
    options: ['O(N)', 'O(1) - single CPU instruction', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Basic bitwise operations (AND, OR, XOR, NOT, shifts) are O(1) - single CPU instruction regardless of value. Very fast.',
  },
  {
    id: 'mc2',
    question: 'What is the complexity of counting set bits in a number?',
    options: [
      'O(1)',
      'O(log N) where N is value - process each bit OR O(k) where k is count of set bits (Kernighan)',
      'O(N)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      "Count set bits: naive O(log N) checks each bit. Brian Kernighan\'s algorithm O(k) where k is number of set bits (n & (n-1) repeatedly). Modern CPUs have O(1) popcount instruction.",
  },
  {
    id: 'mc3',
    question: 'Why is bit manipulation space-efficient?',
    options: [
      'Uses arrays',
      'O(1) space - operations in-place, or compact representation (32 flags in 4 bytes)',
      'Random',
      'Always O(N)',
    ],
    correctAnswer: 1,
    explanation:
      'Bit manipulation typically O(1) space: operations modify in-place without extra structures. Bit sets/masks store data compactly (32 booleans in one 32-bit integer vs 32 bytes).',
  },
  {
    id: 'mc4',
    question:
      'What is the complexity of generating all subsets using bitmasks?',
    options: [
      'O(N)',
      'O(2^N) time, O(1) space per subset - iterate through 2^N masks',
      'O(N²)',
      'O(N log N)',
    ],
    correctAnswer: 1,
    explanation:
      'Generate all subsets: iterate 0 to 2^n-1, each number is bitmask. O(2^N) time (exponential), O(1) space per subset. Total output is O(N×2^N) including elements.',
  },
  {
    id: 'mc5',
    question: 'When is bit manipulation NOT faster?',
    options: [
      'Always faster',
      'When code becomes unreadable, or modern compiler optimizes arithmetic anyway',
      'Never fast',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Don't overuse bit manipulation: 1) Modern compilers optimize n*2 to shift anyway, 2) Readability matters (n%2 clearer than n&1 for most), 3) Use in performance-critical sections only. Balance speed with maintainability.",
  },
];
