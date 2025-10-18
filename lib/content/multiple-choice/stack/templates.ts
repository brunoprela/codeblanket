/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What should you check before calling pop() on a stack?',
    options: [
      'If the stack is sorted',
      'If the stack is empty',
      'If the stack is full',
      'The stack size',
    ],
    correctAnswer: 1,
    explanation:
      'Always check if the stack is empty before popping to avoid errors. Attempting to pop from an empty stack raises an IndexError in Python.',
  },
  {
    id: 'mc2',
    question:
      'In a monotonic stack template, what determines whether to pop elements?',
    options: [
      'The stack is full',
      'Comparing current element with stack top violates monotonic property',
      'Random chance',
      'The stack size exceeds n',
    ],
    correctAnswer: 1,
    explanation:
      'You pop elements when the current element violates the monotonic property (e.g., current is greater than stack top in a decreasing monotonic stack).',
  },
  {
    id: 'mc3',
    question: 'When should you store indices in the stack instead of values?',
    options: [
      'Always store indices',
      'When you need to calculate positions, distances, or widths',
      'When values are too large',
      'Never store indices',
    ],
    correctAnswer: 1,
    explanation:
      'Store indices when you need to calculate positions (for filling result arrays), distances, or widths (like in histogram problems). Store values when you only need comparisons.',
  },
  {
    id: 'mc4',
    question:
      'What is the correct way to implement peek() in Python using a list?',
    options: ['stack[0]', 'stack[-1]', 'stack.peek()', 'stack.top()'],
    correctAnswer: 1,
    explanation:
      "In Python, stack[-1] accesses the last element (top of stack) without removing it. Lists don't have a built-in peek() method.",
  },
  {
    id: 'mc5',
    question: 'In the min stack template, when do you push to the min stack?',
    options: [
      'Only when pushing a new minimum',
      'Every time you push to the main stack',
      'Only at the beginning',
      'Never, it auto-updates',
    ],
    correctAnswer: 1,
    explanation:
      'You push to the min stack every time you push to the main stack, storing the minimum value at that level. This maintains O(1) getMin() at all stack states.',
  },
];
