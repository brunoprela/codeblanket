/**
 * Multiple choice questions for Common Stack Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'In the matching pairs pattern, what do you push onto the stack?',
    options: [
      'Closing brackets',
      'Opening brackets',
      'All characters',
      'Only matched pairs',
    ],
    correctAnswer: 1,
    explanation:
      'You push opening brackets onto the stack. When you encounter a closing bracket, you pop from the stack to check if it matches the most recent opening bracket.',
  },
  {
    id: 'mc2',
    question: 'What is a monotonic stack?',
    options: [
      'A stack that only stores one type of element',
      'A stack that maintains elements in increasing or decreasing order',
      'A stack with fixed size',
      'A stack that never empties',
    ],
    correctAnswer: 1,
    explanation:
      "A monotonic stack maintains elements in either increasing or decreasing order, popping elements that violate this property. It\'s used for problems like finding the next greater element.",
  },
  {
    id: 'mc3',
    question:
      'For the next greater element problem, what type of monotonic stack do you use?',
    options: [
      'Increasing stack',
      'Decreasing stack',
      'Random stack',
      'Sorted stack',
    ],
    correctAnswer: 1,
    explanation:
      'Use a decreasing (or non-increasing) monotonic stack. When you find an element larger than the stack top, you pop elements - they have found their next greater element.',
  },
  {
    id: 'mc4',
    question: 'How does a min stack achieve O(1) getMin() operation?',
    options: [
      'By sorting the stack',
      'By maintaining a separate stack tracking minimum at each level',
      'By using binary search',
      'By keeping the minimum at the bottom',
    ],
    correctAnswer: 1,
    explanation:
      'A min stack uses two stacks: one for values and one for tracking the minimum at each level. When you push/pop from the main stack, you also push/pop from the min stack.',
  },
  {
    id: 'mc5',
    question:
      'What is the time complexity of the next greater element algorithm using a monotonic stack?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'The time complexity is O(n) because each element is pushed onto and popped from the stack at most once, resulting in a single pass through the array.',
  },
];
