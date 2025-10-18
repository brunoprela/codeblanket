/**
 * Multiple choice questions for Introduction to Bit Manipulation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is bit manipulation and why is it efficient?',
    options: [
      'Random operations',
      'Direct operations on binary representation - O(1) CPU instructions, space-efficient',
      'String manipulation',
      'Always slow',
    ],
    correctAnswer: 1,
    explanation:
      'Bit manipulation works directly on binary bits using bitwise operators. Efficient because: O(1) single CPU instructions, O(1) space, can store multiple flags in one integer. Used in permissions, compression, crypto.',
  },
  {
    id: 'mc2',
    question: 'When should you use bit manipulation over regular operations?',
    options: [
      'Always',
      'Performance-critical code, checking even/odd, power of 2, flags, multiply/divide by 2^n',
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use bit manipulation when: 1) Performance-critical (tight loops), 2) Check even/odd (n & 1 vs n % 2), 3) Check power of 2, 4) Store flags compactly, 5) Multiply/divide by powers of 2 (shift). Balance with readability.',
  },
  {
    id: 'mc3',
    question: 'What real-world applications use bit manipulation?',
    options: [
      'None',
      'Unix permissions, RGB colors, IP masks, compression, cryptography, embedded systems',
      'Only academic',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Real-world uses: Unix permissions (rwx=111), RGB colors (24-bit encoding), IP subnet masks, image/video compression, cryptographic algorithms, embedded systems (limited memory), database indexes.',
  },
  {
    id: 'mc4',
    question: 'How can you store multiple boolean flags efficiently?',
    options: [
      'Array of booleans',
      'Single integer with each bit as flag - 32 bits = 32 flags in 4 bytes',
      'Random',
      'Multiple variables',
    ],
    correctAnswer: 1,
    explanation:
      'Store flags in bits of single integer: bit 0 = flag1, bit 1 = flag2, etc. 32-bit int holds 32 flags in 4 bytes vs 32 bytes for 32 bool variables. Set: x |= (1<<i), check: x & (1<<i).',
  },
  {
    id: 'mc5',
    question: 'What is the time complexity of most bitwise operations?',
    options: ['O(N)', 'O(1) - single CPU instruction', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Bitwise operations (AND, OR, XOR, NOT, shifts) are O(1) - single CPU instruction. Some complex operations like counting all set bits is O(log N) where N is value. Very fast compared to arithmetic.',
  },
];
