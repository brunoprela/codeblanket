/**
 * Multiple choice questions for Segment Tree Structure section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const structureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How is a Segment Tree typically stored?',
    options: [
      'Linked nodes',
      'Array of size 4N with tree[1] as root, children at 2i and 2i+1',
      'Hash map',
      'Stack',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree uses array representation: tree[1] is root, for node i, left child is 2i, right child is 2i+1. Size 4N ensures space for complete binary tree.',
  },
  {
    id: 'mc2',
    question: 'What does each node in a Segment Tree represent?',
    options: [
      'Single element',
      'An interval [L, R] with aggregate value (sum/min/max) for that range',
      'Random value',
      'Index',
    ],
    correctAnswer: 1,
    explanation:
      'Each node stores: 1) Interval [L, R] it represents, 2) Aggregate value (sum/min/max) for that range. Leaf nodes are single elements [i, i]. Internal nodes combine children intervals.',
  },
  {
    id: 'mc3',
    question: 'What is the height of a Segment Tree for array of size N?',
    options: ['O(N)', 'O(log N) - complete binary tree', 'O(√N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Segment Tree is complete binary tree, so height is O(log N). Each level doubles nodes until reaching N leaves. This log height enables O(log N) operations.',
  },
  {
    id: 'mc4',
    question: 'How do you find children of node i in array representation?',
    options: [
      'i+1, i+2',
      'Left child: 2i, Right child: 2i+1',
      'Random',
      'i-1, i-2',
    ],
    correctAnswer: 1,
    explanation:
      'Array heap property: for node at index i, left child at 2i, right child at 2i+1, parent at i//2. This allows O(1) navigation without pointers.',
  },
  {
    id: 'mc5',
    question: 'Why do we allocate 4N space for Segment Tree?',
    options: [
      'Random choice',
      'Ensures space for complete binary tree - worst case when N not power of 2',
      'Always need exactly 4N',
      'Optimization',
    ],
    correctAnswer: 1,
    explanation:
      'Complete binary tree with N leaves has at most 2N-1 nodes. For safety with any N (not just powers of 2) and simple indexing, 4N guarantees sufficient space.',
  },
];
