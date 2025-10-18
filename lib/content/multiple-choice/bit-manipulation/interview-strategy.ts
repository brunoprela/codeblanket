/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a bit manipulation problem?',
    options: [
      'Sorting',
      'Single number, missing, duplicate, power of 2, subset, flags, XOR, binary',
      'Shortest path',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Bit manipulation keywords: "single number", "appears once/twice", "missing number", "power of 2", "subsets", "binary representation", "flags", "XOR". Often involves finding patterns or duplicates.',
  },
  {
    id: 'mc2',
    question: 'When should you consider bit manipulation in an interview?',
    options: [
      'Always',
      'O(1) space required, finding duplicates/singles, power of 2, performance-critical, subset generation',
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Consider bit manipulation when: 1) O(1) space required (vs hash map), 2) Finding single/missing numbers (XOR tricks), 3) Power of 2 checks, 4) Need extreme performance, 5) Subset/combination generation.',
  },
  {
    id: 'mc3',
    question: 'What should you clarify in a bit manipulation interview?',
    options: [
      'Nothing',
      'Integer size (32/64-bit)? Signed/unsigned? Negative numbers? Overflow concerns?',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      "Clarify: 1) Integer size (32-bit vs 64-bit affects shift limits), 2) Signed or unsigned (affects right shift behavior), 3) Can numbers be negative (two's complement), 4) Overflow handling.",
  },
  {
    id: 'mc4',
    question: 'What is a common bit manipulation mistake?',
    options: [
      'Using operators',
      'Operator precedence (forgetting parentheses), off-by-one in shifts, sign extension issues',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Precedence: x & 1 == 0 wrong, need (x & 1) == 0, 2) Off-by-one: 1 << 32 undefined (should be 1 << 31 for MSB), 3) Sign extension on right shift of negatives.',
  },
  {
    id: 'mc5',
    question: 'How should you communicate your bit manipulation solution?',
    options: [
      'Just code',
      'Explain the bit pattern, why technique works, walk through example with binary, mention edge cases',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Explain bit pattern/trick (e.g., XOR cancels pairs), 2) Why it works (properties of operators), 3) Walk through example showing binary representation, 4) Edge cases (0, negatives, overflow), 5) Complexity.',
  },
];
