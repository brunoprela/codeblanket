/**
 * Multiple choice questions for Problem-Solving Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the frequency counting pattern best used for?',
    options: [
      'Sorting elements',
      'Counting occurrences of elements',
      'Binary search',
      'Reversing arrays',
    ],
    correctAnswer: 1,
    explanation:
      'The frequency counting pattern uses a hash map to count how many times each element appears, useful for problems like finding most frequent element or checking anagrams.',
  },
  {
    id: 'mc2',
    question:
      'In the complement lookup pattern (two sum), what do you store in the hash table?',
    options: [
      'The target value',
      'All possible sums',
      'Elements seen so far (with their indices)',
      'The largest element',
    ],
    correctAnswer: 2,
    explanation:
      "The complement pattern stores elements you've seen so far in the hash table. For each new element, you check if its complement (target - current) exists in the table.",
  },
  {
    id: 'mc3',
    question:
      'When grouping anagrams, what should be used as the hash map key?',
    options: [
      'The first word in each group',
      'The sorted characters or character count signature',
      'The length of the word',
      'Random numbers',
    ],
    correctAnswer: 1,
    explanation:
      'All anagrams share the same sorted characters (e.g., "eat", "tea", "ate" → "aet") or character counts, making it a perfect key to group them together.',
  },
  {
    id: 'mc4',
    question:
      'What is the time complexity of the deduplication pattern using a hash set?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'The deduplication pattern iterates through elements once, checking each against the hash set in O(1), resulting in O(n) total time instead of O(n²) with nested loops.',
  },
  {
    id: 'mc5',
    question: 'Why is defaultdict useful for the grouping pattern?',
    options: [
      'It sorts keys automatically',
      'It automatically initializes missing keys with a default value',
      'It uses less memory',
      'It is faster than regular dict',
    ],
    correctAnswer: 1,
    explanation:
      'defaultdict automatically creates a default value (like an empty list) for missing keys, eliminating the need to check if a key exists before appending to it.',
  },
];
