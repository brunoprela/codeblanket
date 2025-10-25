/**
 * Multiple choice questions for List Comprehensions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const listcomprehensionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the syntax for a basic list comprehension?',
    options: [
      '[x for x in iterable]',
      '{x for x in iterable}',
      '(x for x in iterable)',
      'list (x in iterable)',
    ],
    correctAnswer: 0,
    explanation:
      'List comprehensions use square brackets []: [expression for item in iterable].',
  },
  {
    id: 'mc2',
    question:
      'How do you add a condition to filter items in a list comprehension?',
    options: [
      '[x where x > 0]',
      '[x for x in nums if x > 0]',
      '[x | x > 0 for x in nums]',
      '[x for x if x > 0 in nums]',
    ],
    correctAnswer: 1,
    explanation: 'Add an if clause at the end: [x for x in nums if condition].',
  },
  {
    id: 'mc3',
    question: 'What does (x**2 for x in range(10)) create?',
    options: ['A list', 'A tuple', 'A generator', 'A set'],
    correctAnswer: 2,
    explanation:
      'Parentheses () create a generator expression, which evaluates lazily. Use [] for lists.',
  },
  {
    id: 'mc4',
    question: 'When should you avoid list comprehensions?',
    options: [
      'Never, always use them',
      'When the logic becomes too complex',
      'When working with numbers',
      'When the list is small',
    ],
    correctAnswer: 1,
    explanation:
      'Avoid list comprehensions when they become too complex or nested. Use regular loops for better readability.',
  },
  {
    id: 'mc5',
    question: 'What is a dictionary comprehension?',
    options: [
      '[key: value for item in iterable]',
      '{key: value for item in iterable}',
      '(key, value for item in iterable)',
      'dict (key=value for item in iterable)',
    ],
    correctAnswer: 1,
    explanation:
      'Dictionary comprehensions use curly braces {}: {key: value for item in iterable}.',
  },
];
