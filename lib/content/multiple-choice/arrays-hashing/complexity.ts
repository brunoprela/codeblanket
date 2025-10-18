/**
 * Multiple choice questions for Time & Space Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of checking if an element exists in an unsorted array?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'Checking existence in an unsorted array requires linear search, which is O(n) since we may need to check every element. A hash set would reduce this to O(1).',
  },
  {
    id: 'mc2',
    question:
      'When comparing arrays vs hash tables for removing duplicates, what is the time complexity difference?',
    options: [
      'Both are O(n)',
      'Array: O(n), Hash table: O(n²)',
      'Array: O(n²), Hash table: O(n)',
      'Array: O(n log n), Hash table: O(log n)',
    ],
    correctAnswer: 2,
    explanation:
      'Without a hash table, checking if each element is a duplicate requires scanning previous elements (O(n²)). With a hash set, we get O(1) lookups for O(n) total time.',
  },
  {
    id: 'mc3',
    question:
      'What is the space complexity overhead difference between arrays and hash tables?',
    options: [
      'Hash tables use less space',
      'They use the same space',
      'Hash tables use more space due to hash structure overhead',
      'Arrays use more space',
    ],
    correctAnswer: 2,
    explanation:
      'Hash tables require additional memory for the hash structure, pointers, and maintaining load factor. Arrays store elements contiguously with minimal overhead.',
  },
  {
    id: 'mc4',
    question:
      'For a problem requiring both fast lookups and maintaining order, which combination is best?',
    options: [
      'Just use an array',
      'Just use a hash table',
      'Use both: hash table for lookups, array for order',
      'Use a set',
    ],
    correctAnswer: 2,
    explanation:
      'When you need both fast lookups (O(1)) and ordered access, use a hash table for lookups and maintain a separate array or list for the ordered elements.',
  },
  {
    id: 'mc5',
    question:
      'When should you prefer sorting an array over using a hash table?',
    options: [
      'When you need O(1) lookups',
      'When memory is extremely limited and O(n log n) time is acceptable',
      'When counting frequencies',
      'When finding duplicates quickly',
    ],
    correctAnswer: 1,
    explanation:
      'Sorting (O(n log n) time, O(1) space with in-place sort) is preferable when memory is severely constrained and the slower time complexity is acceptable.',
  },
];
