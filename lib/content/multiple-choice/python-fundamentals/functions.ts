/**
 * Multiple choice questions for Functions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const functionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pf-functions-mc-1',
    question:
      'What is the output?\n\ndef multiply(a, b=2):\n    return a * b\n\nprint(multiply(5))',
    options: ['10', '5', '7', 'Error'],
    correctAnswer: 0,
    explanation: 'b defaults to 2, so 5 * 2 = 10.',
  },
  {
    id: 'pf-functions-mc-2',
    question: 'What does a function return if no return statement is used?',
    options: ['0', 'Empty string', 'None', 'Error'],
    correctAnswer: 2,
    explanation: 'Functions without a return statement implicitly return None.',
  },
  {
    id: 'pf-functions-mc-3',
    question: 'What does *args allow you to do?',
    options: [
      'Pass a variable number of keyword arguments',
      'Pass a variable number of positional arguments',
      'Make an argument optional',
      'Pass arguments by reference',
    ],
    correctAnswer: 1,
    explanation:
      '*args collects variable numbers of positional arguments into a tuple.',
  },
  {
    id: 'pf-functions-mc-4',
    question:
      'Which is the correct syntax for a lambda function that squares a number?',
    options: [
      'lambda x: x ** 2',
      'lambda(x): x ** 2',
      'def lambda x: x ** 2',
      'lambda x => x ** 2',
    ],
    correctAnswer: 0,
    explanation: 'Lambda syntax is: lambda arguments: expression',
  },
  {
    id: 'pf-functions-mc-5',
    question: 'What is the scope of a variable defined inside a function?',
    options: [
      'Global scope',
      'Local scope (function only)',
      'Module scope',
      'Class scope',
    ],
    correctAnswer: 1,
    explanation:
      'Variables defined inside a function are local to that function unless declared global.',
  },
];
