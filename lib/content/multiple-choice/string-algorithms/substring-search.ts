/**
 * Multiple choice questions for Substring Search & Pattern Matching section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const substringsearchMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-sub1',
    question: 'What is the time complexity of KMP pattern matching algorithm?',
    options: ['O(n)', 'O(m)', 'O(n + m)', 'O(nm)'],
    correctAnswer: 2,
    explanation:
      'KMP guarantees O(n+m) time: O(m) to build LPS array and O(n) to search the text. Each character in text is examined at most once.',
  },
  {
    id: 'mc-sub2',
    question: 'What advantage does Rabin-Karp have over naive string matching?',
    options: [
      'Guaranteed O(n+m) worst case',
      'Can search for multiple patterns simultaneously',
      'Uses less space',
      'Simpler to implement',
    ],
    correctAnswer: 1,
    explanation:
      'Rabin-Karp can search for multiple patterns by computing hashes for all patterns and checking against each substring hash. This is efficient for multi-pattern matching.',
  },
  {
    id: 'mc-sub3',
    question:
      'In Rabin-Karp, why do we need to verify matches after hash collision?',
    options: [
      'To reduce time complexity',
      'Because different strings can have the same hash value',
      'To save memory',
      'Because rolling hash is inaccurate',
    ],
    correctAnswer: 1,
    explanation:
      'Hash collisions occur when different strings produce the same hash value. We must verify character-by-character to confirm a true match. This is called "spurious hit" handling.',
  },
  {
    id: 'mc-sub4',
    question: 'What is the purpose of the LPS array in KMP algorithm?',
    options: [
      'To store all pattern occurrences',
      'To skip redundant comparisons after a mismatch',
      'To calculate pattern hash',
      'To store text positions',
    ],
    correctAnswer: 1,
    explanation:
      'The LPS (Longest Proper Prefix which is also Suffix) array tells us how many characters to skip when a mismatch occurs, avoiding the need to restart pattern matching from the beginning.',
  },
  {
    id: 'mc-sub5',
    question:
      'Which substring search algorithm is best for searching multiple patterns in the same text?',
    options: ['Naive search', 'KMP', 'Rabin-Karp', 'Binary search'],
    correctAnswer: 2,
    explanation:
      'Rabin-Karp is most efficient for multiple patterns because you can compute all pattern hashes once, then check each text substring hash against all pattern hashes in one pass.',
  },
];
