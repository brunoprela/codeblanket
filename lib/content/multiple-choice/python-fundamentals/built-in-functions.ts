/**
 * Multiple choice questions for Essential Built-in Functions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const builtinfunctionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does enumerate() return?',
    options: [
      'Just indices',
      'Just values',
      'Tuples of (index, value)',
      'A dictionary',
    ],
    correctAnswer: 2,
    explanation:
      'enumerate() returns tuples of (index, value) for each item in an iterable.',
  },
  {
    id: 'mc2',
    question: 'What does zip() do with two lists?',
    options: [
      'Compresses them',
      'Pairs corresponding elements',
      'Combines into one list',
      'Sorts both',
    ],
    correctAnswer: 1,
    explanation:
      'zip() pairs corresponding elements: zip([1,2], ["a","b",]) → [(1,"a"), (2,"b")]',
  },
  {
    id: 'mc3',
    question: 'What is the difference between any() and all()?',
    options: [
      'No difference',
      'any() if ANY True, all() if ALL True',
      'any() is faster',
      'all() works with numbers only',
    ],
    correctAnswer: 1,
    explanation:
      'any() returns True if at least one element is truthy; all() only if all are truthy.',
  },
  {
    id: 'mc4',
    question: 'What does map(func, iterable) return?',
    options: ['A list', 'A map object (iterator)', 'A tuple', 'A set'],
    correctAnswer: 1,
    explanation:
      'map() returns a map object (iterator). Use list(map(...)) to get a list.',
  },
  {
    id: 'mc5',
    question: 'How does isinstance() differ from type()?',
    options: [
      'No difference',
      'isinstance() checks class hierarchy, type() exact type',
      'isinstance() deprecated',
      'type() is faster',
    ],
    correctAnswer: 1,
    explanation:
      'isinstance(obj, Class) checks class hierarchy. type(obj) == Class checks exact type only.',
  },
];
