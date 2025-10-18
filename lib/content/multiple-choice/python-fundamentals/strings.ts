/**
 * Multiple choice questions for String Operations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const stringsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pf-strings-mc-1',
    question:
      'What is the output?\n\ntext = "hello world"\nprint(text.title())',
    options: [
      '"Hello World"',
      '"HELLO WORLD"',
      '"hello world"',
      '"Hello world"',
    ],
    correctAnswer: 0,
    explanation: 'title() capitalizes the first letter of each word.',
  },
  {
    id: 'pf-strings-mc-2',
    question: 'What does "Python"[::-1] return?',
    options: ['"Python"', '"nohtyP"', '"Pytho"', 'Error'],
    correctAnswer: 1,
    explanation: '[::-1] reverses the string using negative step.',
  },
  {
    id: 'pf-strings-mc-3',
    question: 'Which method checks if a string contains only digits?',
    options: ['isnum()', 'isdigit()', 'isnumber()', 'isint()'],
    correctAnswer: 1,
    explanation: 'isdigit() returns True if all characters are digits.',
  },
  {
    id: 'pf-strings-mc-4',
    question: 'What is the result of: "hello" + " " + "world"?',
    options: ['"hello world"', '"helloworld"', '"hello  world"', 'Error'],
    correctAnswer: 0,
    explanation: 'String concatenation with + joins strings together.',
  },
  {
    id: 'pf-strings-mc-5',
    question: 'What does "abc" * 3 produce?',
    options: ['"abcabcabc"', '"abc3"', '["abc", "abc", "abc"]', 'Error'],
    correctAnswer: 0,
    explanation: 'String multiplication repeats the string n times.',
  },
];
