/**
 * Multiple choice questions for Common Recursion Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How many recursive calls does Fibonacci(5) make with naive recursion?',
    options: ['5 calls', '10 calls', '15 calls', '31 calls'],
    correctAnswer: 2,
    explanation:
      'Naive Fibonacci makes exponential calls: fib(5) = 15 total calls. Each call branches into two more (except base cases), creating a binary tree of calls.',
  },
  {
    id: 'mc2',
    question: 'What is tail recursion?',
    options: [
      'Recursion at the end of a program',
      'When the recursive call is the last operation in the function',
      'Recursion with two calls',
      'Recursion without a base case',
    ],
    correctAnswer: 1,
    explanation:
      'Tail recursion occurs when the recursive call is the final operation with no pending computations. This allows some compilers to optimize it into iteration.',
  },
  {
    id: 'mc3',
    question: 'Why is naive Fibonacci recursion inefficient?',
    options: [
      'It uses too much memory',
      'It recalculates the same values multiple times',
      'It has too many base cases',
      'It is not a valid recursive algorithm',
    ],
    correctAnswer: 1,
    explanation:
      'Naive Fibonacci recalculates the same values repeatedly. For example, fib(5) calls fib(3) twice, and each fib(3) recalculates fib(2), fib(1), etc., leading to exponential time.',
  },
  {
    id: 'mc4',
    question: 'What is the time complexity of naive recursive Fibonacci?',
    options: ['O(n)', 'O(n log n)', 'O(2^n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'Naive Fibonacci has O(2^n) time complexity because each call branches into two more calls, creating an exponential tree of recursive calls.',
  },
  {
    id: 'mc5',
    question: 'What is indirect recursion?',
    options: [
      'When a function calls itself through another function',
      'When recursion uses iteration internally',
      'When the base case is indirect',
      'When multiple functions call themselves',
    ],
    correctAnswer: 0,
    explanation:
      'Indirect recursion is when function A calls function B, and B calls A back. This creates a cycle: A → B → A → B...',
  },
];
