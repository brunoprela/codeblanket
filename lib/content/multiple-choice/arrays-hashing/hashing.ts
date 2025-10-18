/**
 * Multiple choice questions for Hash Tables: Fast Lookups & Storage section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hashingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the average-case time complexity for hash table insert, delete, and search operations?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 0,
    explanation:
      'Hash tables provide O(1) average-case complexity for insert, delete, and search operations through direct index computation using a hash function.',
  },
  {
    id: 'mc2',
    question:
      'What causes hash table operations to degrade to O(n) worst case?',
    options: [
      'The table is too small',
      'Many hash collisions occur',
      'The keys are not sorted',
      'The table is empty',
    ],
    correctAnswer: 1,
    explanation:
      'When many keys hash to the same index (collisions), the hash table degenerates to a linked list in that bucket, causing O(n) worst-case lookup time.',
  },
  {
    id: 'mc3',
    question:
      'In Python, which is the best data structure for grouping elements with automatic list initialization?',
    options: ['dict', 'set', 'defaultdict', 'Counter'],
    correctAnswer: 2,
    explanation:
      'defaultdict from collections automatically initializes missing keys with a default value (like an empty list), preventing KeyError and simplifying grouping logic.',
  },
  {
    id: 'mc4',
    question: 'What is the purpose of the load factor in a hash table?',
    options: [
      'To determine the hash function',
      'To determine when to resize the table',
      'To count the number of elements',
      'To sort the elements',
    ],
    correctAnswer: 1,
    explanation:
      'The load factor (typically 0.75) determines when the hash table should resize. When load factor is exceeded, the table resizes to maintain O(1) operations.',
  },
  {
    id: 'mc5',
    question:
      'For finding the first non-repeating character in a string, what is the optimal approach?',
    options: [
      'Sort the string',
      'Use nested loops',
      'Two passes with hash map: count frequencies, then find first with count 1',
      'Use binary search',
    ],
    correctAnswer: 2,
    explanation:
      'Two passes with a hash map is optimal: first pass counts character frequencies in O(n), second pass checks counts in order to find first character with count 1, total O(n) time.',
  },
];
