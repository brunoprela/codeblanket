/**
 * Multiple choice questions for Generators & Iterators section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const generatorsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the memory complexity of a generator that yields n items?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n^2)'],
    correctAnswer: 0,
    explanation:
      'Generators have O(1) space complexity because they produce values one at a time without storing all values in memory, regardless of how many items they yield.',
  },
  {
    id: 'mc2',
    question: 'Which syntax creates a generator expression?',
    options: [
      '[x**2 for x in range(10)]',
      '{x**2 for x in range(10)}',
      '(x**2 for x in range(10))',
      'x**2 for x in range(10)',
    ],
    correctAnswer: 2,
    explanation:
      'Generator expressions use parentheses (). Square brackets create lists, curly braces create sets, and the last option is invalid syntax.',
  },
  {
    id: 'mc3',
    question: 'What happens when a generator function is called?',
    options: [
      'The function executes immediately',
      'It returns a generator object without executing the function body',
      'It raises an error',
      'It returns None',
    ],
    correctAnswer: 1,
    explanation:
      'Calling a generator function returns a generator object without executing the function body. The code runs only when you iterate over the generator or call next().',
  },
  {
    id: 'mc4',
    question: 'Can generators represent infinite sequences?',
    options: [
      'No, they must be finite',
      'Yes, because they produce values lazily',
      'Only with special syntax',
      'Yes, but they crash after 1 million items',
    ],
    correctAnswer: 1,
    explanation:
      'Generators can represent infinite sequences because they produce values lazily on-demand, never storing the entire sequence. Example: while True: yield value',
  },
  {
    id: 'mc5',
    question: 'What is the main benefit of using generator pipelines?',
    options: [
      'Faster execution',
      'Memory efficiency through lazy evaluation',
      'Automatic parallelization',
      'Better error handling',
    ],
    correctAnswer: 1,
    explanation:
      'Generator pipelines keep memory usage constant by processing one item through all stages before moving to the next, avoiding intermediate result storage.',
  },
];
