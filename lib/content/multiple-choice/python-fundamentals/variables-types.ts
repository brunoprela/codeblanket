/**
 * Multiple choice questions for Variables and Data Types section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const variablestypesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pf-variables-mc-1',
    question:
      'What does the following code output?\n\nx = "5"\ny = 2\nprint(x * y)',
    options: ['10', '7', '"55"', '"52"'],
    correctAnswer: 2,
    explanation: 'String multiplication repeats the string. "5" * 2 = "55".',
  },
  {
    id: 'pf-variables-mc-2',
    question:
      'Which statement correctly converts the string "123" to an integer?',
    options: ['integer("123")', 'int("123")', 'toInt("123")', 'Number("123")'],
    correctAnswer: 1,
    explanation: 'The int() function converts strings to integers in Python.',
  },
  {
    id: 'pf-variables-mc-3',
    question: 'What is the result of: type(3.0) == type(3)?',
    options: ['True', 'False', 'TypeError', '3.0'],
    correctAnswer: 1,
    explanation: '3.0 is a float and 3 is an int, so they are different types.',
  },
  {
    id: 'pf-variables-mc-4',
    question: 'Which of the following is NOT a valid variable name?',
    options: ['my_var', '_private', '2fast', 'myVar2'],
    correctAnswer: 2,
    explanation:
      'Variable names cannot start with a number. "2fast" is invalid.',
  },
  {
    id: 'pf-variables-mc-5',
    question: 'What does bool("") evaluate to?',
    options: ['True', 'False', 'None', 'Error'],
    correctAnswer: 1,
    explanation:
      'Empty strings are falsy in Python, so bool("") returns False.',
  },
];
