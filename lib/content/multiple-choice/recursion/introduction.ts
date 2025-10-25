/**
 * Multiple choice questions for Introduction to Recursion section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of calculating factorial (n) recursively?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'Factorial recursion makes exactly n function calls (for n, n-1, n-2, ..., 1), each doing O(1) work. Total: O(n) time complexity.',
  },
  {
    id: 'mc2',
    question: 'What happens if a recursive function lacks a base case?',
    options: [
      'It returns None',
      'It throws a syntax error',
      'It causes infinite recursion and stack overflow',
      'It returns 0',
    ],
    correctAnswer: 2,
    explanation:
      'Without a base case, the function calls itself indefinitely, creating stack frames until the stack overflows, resulting in a RecursionError in Python.',
  },
  {
    id: 'mc3',
    question: 'Which statement about the call stack in recursion is TRUE?',
    options: [
      'The call stack is emptied before recursion starts',
      'Stack frames are added in FIFO order',
      'Each recursive call creates a new stack frame',
      'The stack remains constant size during recursion',
    ],
    correctAnswer: 2,
    explanation:
      'Each recursive call pushes a new stack frame onto the call stack, storing local variables and return address. Frames are popped in LIFO order.',
  },
  {
    id: 'mc4',
    question:
      'What is the space complexity of factorial (n) recursive implementation?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'The recursive call stack holds n frames simultaneously (one for each call from n down to 1), requiring O(n) space.',
  },
  {
    id: 'mc5',
    question: 'In recursive functions, what does "unwinding" refer to?',
    options: [
      'Removing loops from the code',
      'Converting recursion to iteration',
      'Returning values as stack frames pop off',
      'Simplifying the base case',
    ],
    correctAnswer: 2,
    explanation:
      'Unwinding is the process where stack frames pop off after reaching the base case, with each call returning its result to its caller.',
  },
];
