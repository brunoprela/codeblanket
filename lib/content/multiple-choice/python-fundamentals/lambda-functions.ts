/**
 * Multiple choice questions for Lambda Functions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const lambdafunctionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a lambda function?',
    options: [
      'A regular function',
      'An anonymous single-expression function',
      'A class method',
      'A module',
    ],
    correctAnswer: 1,
    explanation:
      'Lambda functions are anonymous (unnamed) functions defined in a single expression using the lambda keyword.',
  },
  {
    id: 'mc2',
    question: 'What is the syntax for a lambda function?',
    options: [
      'lambda x: x*2',
      'def lambda (x): x*2',
      'function (x) => x*2',
      'x => x*2',
    ],
    correctAnswer: 0,
    explanation:
      'Lambda syntax: lambda parameters: expression. For example: lambda x: x*2',
  },
  {
    id: 'mc3',
    question: 'Can a lambda function contain multiple statements?',
    options: [
      'Yes, any number',
      'Yes, up to 10',
      'No, only single expression',
      'Only if using semicolons',
    ],
    correctAnswer: 2,
    explanation:
      'Lambda functions can only contain a single expression, not statements. For complex logic, use regular functions.',
  },
  {
    id: 'mc4',
    question: 'Which function commonly uses lambda as a parameter?',
    options: ['print()', 'sorted()', 'len()', 'input()'],
    correctAnswer: 1,
    explanation:
      'sorted(), map(), filter(), reduce() commonly use lambda functions for key/operation parameters.',
  },
  {
    id: 'mc5',
    question: 'When should you avoid lambda functions?',
    options: [
      'With sorted()',
      'For complex multi-step logic',
      'With map()',
      'For simple operations',
    ],
    correctAnswer: 1,
    explanation:
      'Avoid lambdas for complex logic - use regular named functions for readability and debuggability.',
  },
];
