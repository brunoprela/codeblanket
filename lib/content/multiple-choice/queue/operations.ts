/**
 * Multiple choice questions for Queue Operations & Implementation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const operationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of enqueue and dequeue in a properly implemented queue?',
    options: [
      'Both O(1)',
      'Both O(n)',
      'Enqueue O(1), Dequeue O(n)',
      'Enqueue O(n), Dequeue O(1)',
    ],
    correctAnswer: 0,
    explanation:
      'A properly implemented queue (using deque or linked list) has O(1) time complexity for both enqueue and dequeue operations.',
  },
  {
    id: 'mc2',
    question: 'Why is using a Python list as a queue inefficient?',
    options: [
      'Enqueue (append) is O(n)',
      'Dequeue (pop(0)) is O(n) due to shifting',
      'It uses too much memory',
      'Lists cannot hold queue data',
    ],
    correctAnswer: 1,
    explanation:
      'pop(0) removes the first element and shifts all remaining elements left, making it O(n). Use collections.deque for O(1) operations.',
  },
  {
    id: 'mc3',
    question:
      "What is Python\'s recommended data structure for implementing a queue?",
    options: ['list', 'tuple', 'collections.deque', 'set'],
    correctAnswer: 2,
    explanation:
      'collections.deque (double-ended queue) provides O(1) append and popleft operations, making it ideal for queues.',
  },
  {
    id: 'mc4',
    question: 'How would you implement a queue using two stacks?',
    options: [
      'Use one stack for everything',
      'Push to stack1, pop from stack1',
      'Push to stack1, transfer to stack2 when popping',
      "It\'s impossible",
    ],
    correctAnswer: 2,
    explanation:
      'Push all elements to stack1. When dequeuing, if stack2 is empty, transfer all from stack1 to stack2 (reversing order), then pop from stack2.',
  },
  {
    id: 'mc5',
    question: 'What should happen when you try to dequeue from an empty queue?',
    options: [
      'Return None',
      'Return 0',
      'Raise an exception or return error indicator',
      'Do nothing',
    ],
    correctAnswer: 2,
    explanation:
      'Dequeuing from an empty queue should raise an exception (like IndexError) or return a special error indicator to prevent invalid operations.',
  },
];
