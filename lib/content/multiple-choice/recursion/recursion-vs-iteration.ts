/**
 * Multiple choice questions for Recursion vs Iteration section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const recursionvsiterationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-reciter1',
    question:
      'What is the space complexity of recursive factorial compared to iterative?',
    options: [
      'Both O(1)',
      'Recursive O(n), Iterative O(1)',
      'Recursive O(n), Iterative O(n)',
      'Both O(n)',
    ],
    correctAnswer: 1,
    explanation:
      'Recursive factorial uses O(n) space due to n call stack frames. Iterative factorial uses O(1) space with just a loop variable.',
  },
  {
    id: 'mc-reciter2',
    question: 'Which problem is naturally better suited for recursion?',
    options: [
      'Calculating sum of array elements',
      'Binary tree traversal',
      'Finding maximum in array',
      'Linear search',
    ],
    correctAnswer: 1,
    explanation:
      'Binary tree traversal is naturally recursive - each node has left and right subtrees, making recursive definition elegant. Array problems like sum/max are better iterative.',
  },
  {
    id: 'mc-reciter3',
    question: 'What causes stack overflow in recursive functions?',
    options: [
      'Too many variables',
      'Infinite recursion or too deep recursion',
      'Using global variables',
      'Complex calculations',
    ],
    correctAnswer: 1,
    explanation:
      'Stack overflow occurs when recursion depth exceeds stack limit - either from infinite recursion (missing/wrong base case) or legitimate deep recursion exceeding system limits.',
  },
  {
    id: 'mc-reciter4',
    question: 'What is tail recursion?',
    options: [
      'Recursion at the end of a program',
      'When recursive call is the last operation in the function',
      'Recursion with two base cases',
      'Recursion that returns the tail of a list',
    ],
    correctAnswer: 1,
    explanation:
      'Tail recursion is when the recursive call is the last operation (tail position). Some compilers optimize this to iteration, eliminating stack growth: O(n) space → O(1).',
  },
  {
    id: 'mc-reciter5',
    question: 'Why might iterative solutions be preferred in production code?',
    options: [
      'They are always faster',
      'They are easier to write',
      'They avoid stack overflow and have lower overhead',
      'They use less memory in all cases',
    ],
    correctAnswer: 2,
    explanation:
      "Iterative solutions avoid stack overflow risk and function call overhead, making them more reliable and efficient for production. They're not always easier or using less memory though.",
  },
];
