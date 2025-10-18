/**
 * Multiple choice questions for Advanced Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const advancedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the purpose of using two stacks for expression evaluation?',
    options: [
      'To double the stack size',
      'To separate values and operators and handle precedence correctly',
      'To make the algorithm faster',
      'To avoid using recursion',
    ],
    correctAnswer: 1,
    explanation:
      'Two stacks (one for values, one for operators) allow us to handle operator precedence correctly. We can delay or prioritize operations based on precedence as we scan the expression.',
  },
  {
    id: 'mc2',
    question:
      'In the stock span problem, what property does the stack maintain?',
    options: [
      'Monotonic increasing prices',
      'Monotonic decreasing prices',
      'Balanced prices',
      'Random order',
    ],
    correctAnswer: 1,
    explanation:
      'The stack maintains a monotonic decreasing sequence of prices (represented by indices). We pop prices that are less than or equal to the current price, as they cannot be span boundaries for future days.',
  },
  {
    id: 'mc3',
    question:
      'What is the time complexity of the largest rectangle in histogram problem using a monotonic stack?',
    options: ['O(n²)', 'O(n log n)', 'O(n)', 'O(log n)'],
    correctAnswer: 2,
    explanation:
      'Using a monotonic stack, each bar is pushed and popped at most once, resulting in O(n) time complexity. This is much better than the O(n²) brute force approach.',
  },
  {
    id: 'mc4',
    question:
      'When converting recursive DFS to iterative using a stack, what replaces the call stack?',
    options: [
      'A queue',
      'A heap',
      'An explicit stack data structure',
      'An array',
    ],
    correctAnswer: 2,
    explanation:
      'An explicit stack replaces the implicit call stack. Instead of recursive function calls, we push nodes onto our stack and pop them to process, mimicking the LIFO behavior of recursive calls.',
  },
  {
    id: 'mc5',
    question:
      'In backtracking with stacks, what does each stack element typically represent?',
    options: [
      'A single character',
      'A decision point or state in the solution space',
      'A random number',
      'The final answer',
    ],
    correctAnswer: 1,
    explanation:
      'Each stack element represents a decision point or state (e.g., current string, open/close counts). The stack allows us to explore all possible paths by pushing new states and popping when backtracking.',
  },
];
