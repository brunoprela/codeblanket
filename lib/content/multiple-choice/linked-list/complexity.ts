/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of accessing an element at a specific index in a linked list?',
    options: ['O(1)', 'O(N)', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Accessing an element at index i requires traversing from the head through i nodes, taking O(N) time. Unlike arrays, linked lists do not support O(1) random access.',
  },
  {
    id: 'mc2',
    question:
      'Why is inserting at the tail O(N) in a singly linked list without a tail pointer?',
    options: [
      'Because linked lists are slow',
      'Because you must traverse the entire list to find the last node',
      'Because insertion requires copying all nodes',
      'It is actually O(1)',
    ],
    correctAnswer: 1,
    explanation:
      'Without a tail pointer, you must traverse all N nodes to find the last node before inserting. With a tail pointer, it becomes O(1).',
  },
  {
    id: 'mc3',
    question:
      "What is the space complexity of Floyd\'s cycle detection algorithm?",
    options: ['O(N)', 'O(1)', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      "Floyd\'s algorithm uses only two pointers (slow and fast) regardless of list size, making it O(1) space. This is more efficient than hash set approaches which use O(N) space.",
  },
  {
    id: 'mc4',
    question: 'Why do recursive linked list solutions use O(N) space?',
    options: [
      'They create new nodes',
      'Each recursive call adds a stack frame to the call stack',
      'They use extra arrays',
      "They don't use O(N) space",
    ],
    correctAnswer: 1,
    explanation:
      'Recursive solutions make N recursive calls for N nodes, with each call adding a stack frame to the call stack. This results in O(N) space complexity, unlike iterative solutions which use O(1) space.',
  },
  {
    id: 'mc5',
    question:
      'What is the time complexity of merging K sorted linked lists using a min heap?',
    options: ['O(NK)', 'O(N log K)', 'O(K log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Using a min heap of size K, each of the N total nodes is added and removed from the heap once, with each operation taking O(log K) time, giving O(N log K) total time.',
  },
];
