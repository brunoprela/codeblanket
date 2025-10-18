/**
 * Multiple choice questions for Data Structures section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const datastructuresMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pf-datastructures-mc-1',
    question:
      'What does this list comprehension create?\n\n[x*2 for x in range(5) if x > 2]',
    options: ['[0, 2, 4, 6, 8]', '[6, 8]', '[3, 4]', '[6, 8, 10]'],
    correctAnswer: 1,
    explanation: 'Filters x > 2 (3, 4), then multiplies by 2: [6, 8].',
  },
  {
    id: 'pf-datastructures-mc-2',
    question: 'What is the main difference between lists and tuples?',
    options: [
      'Lists are faster',
      'Tuples are immutable',
      'Lists can only store numbers',
      'Tuples can be nested',
    ],
    correctAnswer: 1,
    explanation:
      'Tuples are immutable (cannot be changed after creation), while lists are mutable.',
  },
  {
    id: 'pf-datastructures-mc-3',
    question: 'How do you create an empty set?',
    options: ['{}', 'set()', '[]', 'Set()'],
    correctAnswer: 1,
    explanation: '{} creates an empty dictionary. Use set() for an empty set.',
  },
  {
    id: 'pf-datastructures-mc-4',
    question:
      'What does this code output?\n\na = {1, 2, 3}\nb = {3, 4, 5}\nprint(a & b)',
    options: ['{1, 2, 3, 4, 5}', '{3}', '{1, 2, 4, 5}', '{}'],
    correctAnswer: 1,
    explanation:
      'The & operator performs set intersection, returning elements common to both sets.',
  },
  {
    id: 'pf-datastructures-mc-5',
    question:
      'Which data structure would be most efficient for checking if an item exists?',
    options: ['List', 'Tuple', 'Set', 'String'],
    correctAnswer: 2,
    explanation:
      'Sets use hash tables, making membership testing O(1) average case, much faster than lists or tuples which are O(n).',
  },
];
