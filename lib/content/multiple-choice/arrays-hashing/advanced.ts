/**
 * Multiple choice questions for Advanced Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'When would you use multiple hash tables instead of one?',
    options: [
      'To save memory',
      'When tracking different relationships or bidirectional mappings',
      'To make the code simpler',
      'Never, one is always enough',
    ],
    correctAnswer: 1,
    explanation:
      'Multiple hash tables are useful when tracking different relationships (like LRU cache needing both key→node and ordering) or bidirectional mappings (name→ID and ID→name).',
  },
  {
    id: 'mc2',
    question: 'What makes an object hashable in Python?',
    options: [
      'It must be a string',
      'It must be immutable and have a consistent hash value',
      'It must be a number',
      'It must be small',
    ],
    correctAnswer: 1,
    explanation:
      'Hashable objects must be immutable and provide a consistent hash value. Mutable objects like lists cannot be hashable because their hash would change when modified.',
  },
  {
    id: 'mc3',
    question: 'What problem does the rolling hash technique solve?',
    options: [
      'Sorting strings',
      'Finding duplicates',
      'Efficiently computing hash values for sliding window of characters',
      'Reversing arrays',
    ],
    correctAnswer: 2,
    explanation:
      'Rolling hash (used in Rabin-Karp) efficiently updates the hash value when sliding a window by removing the leaving character and adding the entering character in O(1).',
  },
  {
    id: 'mc4',
    question: 'What is coordinate compression used for?',
    options: [
      'Reducing file sizes',
      'Mapping large values to small sequential indices',
      'Hashing passwords',
      'Sorting arrays',
    ],
    correctAnswer: 1,
    explanation:
      'Coordinate compression maps a set of potentially large or sparse values to a small dense range (0, 1, 2, ...), useful when values are large but their relative order matters.',
  },
  {
    id: 'mc5',
    question: 'Why are tuples used as hash keys instead of lists?',
    options: [
      'Tuples are faster',
      'Tuples are immutable and therefore hashable',
      'Tuples use less memory',
      'Lists are deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'Tuples are immutable, making them hashable and safe to use as dictionary keys. Lists are mutable and would break hash table invariants if their contents changed.',
  },
];
