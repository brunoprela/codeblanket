/**
 * Multiple choice questions for Queue Variations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const variationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of a circular queue over a regular array-based queue?',
    options: [
      'It is faster',
      'It reuses space from dequeued elements without shifting',
      'It uses less memory',
      'It maintains sorted order',
    ],
    correctAnswer: 1,
    explanation:
      'Circular queues reuse space by wrapping the rear pointer back to the front using modulo arithmetic. This avoids the need to shift elements or grow the array when the rear reaches the end.',
  },
  {
    id: 'mc2',
    question:
      'In a priority queue, what data structure is typically used for implementation?',
    options: ['Array', 'Linked List', 'Heap', 'Hash Table'],
    correctAnswer: 2,
    explanation:
      'Priority queues are typically implemented using a heap (min-heap or max-heap) which provides O(log n) enqueue and dequeue operations based on priority.',
  },
  {
    id: 'mc3',
    question:
      'Which queue variation allows efficient addition and removal from both ends?',
    options: [
      'Circular Queue',
      'Priority Queue',
      'Deque (Double-Ended Queue)',
      'Blocking Queue',
    ],
    correctAnswer: 2,
    explanation:
      'Deque (double-ended queue) supports O(1) operations at both ends: append, appendleft, pop, and popleft.',
  },
  {
    id: 'mc4',
    question:
      'What is the time complexity of dequeue operation in a priority queue implemented with a heap?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)'],
    correctAnswer: 1,
    explanation:
      'Dequeue (removing the highest priority element) in a heap-based priority queue requires removing the root and reheapifying, which takes O(log n) time.',
  },
  {
    id: 'mc5',
    question: 'Which Python module provides a thread-safe blocking queue?',
    options: ['collections', 'queue', 'threading', 'asyncio'],
    correctAnswer: 1,
    explanation:
      'The queue module in Python provides Queue, which is a thread-safe blocking queue useful for producer-consumer patterns in multithreaded applications.',
  },
];
