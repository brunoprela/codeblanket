/**
 * Multiple choice questions for XOR Properties and Applications section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const xorpropertiesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What are the key properties of XOR?',
    options: [
      'None',
      'Commutative, associative, self-inverse (x ^ x = 0), identity (x ^ 0 = x)',
      'Random',
      'Only commutative',
    ],
    correctAnswer: 1,
    explanation:
      'XOR properties: 1) Commutative: a ^ b = b ^ a, 2) Associative: (a ^ b) ^ c = a ^ (b ^ c), 3) Self-inverse: x ^ x = 0, 4) Identity: x ^ 0 = x. These enable finding single number, duplicates.',
  },
  {
    id: 'mc2',
    question:
      'How do you find the single number in array where every other appears twice?',
    options: [
      'Hash map',
      'XOR all elements - pairs cancel (x ^ x = 0), single remains',
      'Sort',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'XOR all elements: a ^ a ^ b ^ b ^ c = (a ^ a) ^ (b ^ b) ^ c = 0 ^ 0 ^ c = c. Pairs cancel due to XOR self-inverse property. O(N) time, O(1) space.',
  },
  {
    id: 'mc3',
    question:
      'How do you find two single numbers when every other appears twice?',
    options: [
      'Cannot do efficiently',
      'XOR all to get x^y, find differing bit, partition and XOR each group',
      'Sort',
      'Hash map',
    ],
    correctAnswer: 1,
    explanation:
      'XOR all gives x^y (pairs cancel). Find any set bit in x^y (x and y differ there). Partition array by that bit, XOR each partition separately to get x and y. O(N) time, O(1) space.',
  },
  {
    id: 'mc4',
    question: 'What is the XOR trick for swapping adjacent bits?',
    options: [
      'Impossible',
      'XOR with pattern alternating 01: x ^ 0b01010101 swaps adjacent pairs',
      'Random',
      'Shift only',
    ],
    correctAnswer: 1,
    explanation:
      'Swap adjacent bits: separate odd and even bits, shift, OR. Or use XOR with alternating pattern. More generally, use bit masking and shifting to rearrange bit patterns.',
  },
  {
    id: 'mc5',
    question: 'How does XOR help detect missing number in array [1..n]?',
    options: [
      'Cannot help',
      'XOR all numbers 1..n with array elements - missing number remains',
      'Sum only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'XOR 1^2^...^n with all array elements. Present numbers cancel out (x ^ x = 0), missing number remains. Alternative to sum formula, handles overflow better. O(N) time, O(1) space.',
  },
];
