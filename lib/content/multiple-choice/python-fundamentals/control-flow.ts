/**
 * Multiple choice questions for Control Flow section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const controlflowMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pf-control-mc-1',
    question:
      'What is the output?\n\nfor i in range(3):\n    if i == 1:\n        continue\n    print(i)',
    options: ['0 1 2', '0 2', '1 2', '0 1'],
    correctAnswer: 1,
    explanation:
      'continue skips the rest of the current iteration, so 1 is not printed.',
  },
  {
    id: 'pf-control-mc-2',
    question: 'What does range(5) produce?',
    options: [
      '[1, 2, 3, 4, 5]',
      '[0, 1, 2, 3, 4]',
      '[0, 1, 2, 3, 4, 5]',
      '(0, 1, 2, 3, 4)',
    ],
    correctAnswer: 1,
    explanation:
      'range(5) generates numbers from 0 up to (but not including) 5.',
  },
  {
    id: 'pf-control-mc-3',
    question: 'What happens when "break" is used in a loop?',
    options: [
      'Skips current iteration',
      'Exits the loop completely',
      'Pauses the loop',
      'Restarts the loop',
    ],
    correctAnswer: 1,
    explanation:
      'break exits the loop immediately, skipping any remaining iterations.',
  },
  {
    id: 'pf-control-mc-4',
    question:
      'What will this code print?\n\nx = 15\nif x > 10:\n    print("A")\nelif x > 5:\n    print("B")\nelse:\n    print("C")',
    options: ['"A"', '"B"', '"C"', '"A" and "B"'],
    correctAnswer: 0,
    explanation:
      'The first condition (x > 10) is true, so "A" is printed and the rest is skipped.',
  },
  {
    id: 'pf-control-mc-5',
    question: 'What is the difference between == and = in Python?',
    options: [
      'No difference',
      '== is comparison, = is assignment',
      '= is comparison, == is assignment',
      'Both are assignment',
    ],
    correctAnswer: 1,
    explanation:
      '== compares values for equality, while = assigns a value to a variable.',
  },
];
