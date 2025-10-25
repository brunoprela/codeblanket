/**
 * Multiple choice questions for BFS on Trees (Level-Order Traversal) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const treebfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is level-order traversal?',
    options: [
      'Preorder traversal',
      'Visit nodes level by level, left to right, using BFS with queue',
      'Inorder traversal',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Level-order traversal visits all nodes at level 0, then level 1, then level 2, etc. Uses BFS with queue. For tree [1,[2,3],[4,5]], visits: [1], [2,3], [4,5].',
  },
  {
    id: 'mc2',
    question: 'How do you get nodes separated by level in BFS?',
    options: [
      'Cannot separate',
      'Track level size before processing: size = queue.length, process exactly size nodes',
      'Use two queues',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Before each level, capture queue size (number of nodes at current level). Process exactly that many nodes, adding their children. Each iteration processes one complete level.',
  },
  {
    id: 'mc3',
    question: 'What is the pattern for level-by-level BFS?',
    options: [
      'Random',
      'While queue: level_size = len (queue), for i in range (level_size): process node, add children',
      'Single loop only',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      "Pattern: outer while loop (levels), inner for loop processing level_size nodes (current level). Inner loop adds next level's nodes. This cleanly separates levels.",
  },
  {
    id: 'mc4',
    question: 'How do you find rightmost node at each level?',
    options: [
      'Cannot do',
      'Level-order BFS, track last node processed in each level iteration',
      'DFS only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Level-order BFS with level tracking. In inner loop processing each level, last node (i == level_size-1) is rightmost. Add to result.',
  },
  {
    id: 'mc5',
    question: 'Why is tree BFS always iterative, not recursive?',
    options: [
      'Can be recursive',
      "Queue-based nature doesn't fit recursion (processes breadth-first); DFS's depth-first fits recursion naturally",
      'Random',
      'Too slow',
    ],
    correctAnswer: 1,
    explanation:
      'BFS requires queue to maintain level-by-level order. Recursion uses stack (LIFO), which gives depth-first order. BFS is naturally iterative. DFS fits recursion because call stack provides LIFO.',
  },
];
