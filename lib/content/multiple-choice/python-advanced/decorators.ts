/**
 * Multiple choice questions for Decorators & Function Wrapping section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const decoratorsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does functools.wraps do?',
    options: [
      'Makes functions run faster',
      'Preserves the original function metadata in the wrapper',
      'Adds error handling to functions',
      'Converts functions to generators',
    ],
    correctAnswer: 1,
    explanation:
      'functools.wraps copies metadata like __name__, __doc__, and __module__ from the original function to the wrapper, preserving the function identity for debugging and introspection.',
  },
  {
    id: 'mc2',
    question:
      'What is the time complexity improvement of using @lru_cache on fibonacci?',
    options: [
      'O(n) to O(1)',
      'O(2^n) to O(n)',
      'O(n^2) to O(n)',
      'O(n!) to O(n^2)',
    ],
    correctAnswer: 1,
    explanation:
      'Without memoization, recursive fibonacci is O(2^n) due to redundant calculations. With @lru_cache, each fibonacci(n) is calculated only once, reducing time complexity to O(n).',
  },
  {
    id: 'mc3',
    question:
      'When stacking decorators like @a @b @c def func(), what is the execution order at runtime?',
    options: [
      'a, b, c, func',
      'c, b, a, func',
      'func, a, b, c',
      'func, c, b, a',
    ],
    correctAnswer: 0,
    explanation:
      'Decorators execute from top to bottom at runtime: a runs first, then b, then c, and finally the original function func.',
  },
  {
    id: 'mc4',
    question:
      'What requirement must function arguments meet to use @lru_cache?',
    options: [
      'Must be strings',
      'Must be hashable (immutable)',
      'Must be integers',
      'No requirements',
    ],
    correctAnswer: 1,
    explanation:
      '@lru_cache stores results in a dictionary keyed by arguments, so arguments must be hashable (immutable types like int, str, tuple).',
  },
  {
    id: 'mc5',
    question: 'What is a common use case for decorators?',
    options: [
      'Adding authentication checks to functions',
      'Sorting lists',
      'Creating loops',
      'Defining variables',
    ],
    correctAnswer: 0,
    explanation:
      'Decorators are commonly used for cross-cutting concerns like authentication, logging, caching, and timing - functionality that applies to multiple functions.',
  },
];
