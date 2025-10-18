/**
 * Multiple choice questions for Introduction to DFS section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the core concept of DFS?',
    options: [
      'Go wide first',
      'Explore as far as possible along each branch before backtracking',
      'Find shortest path',
      'Random traversal',
    ],
    correctAnswer: 1,
    explanation:
      'DFS explores deeply first - follows one path as far as possible before backtracking. Uses stack (recursion or explicit) for LIFO behavior. Goes deep before wide.',
  },
  {
    id: 'mc2',
    question: 'What data structure does DFS use?',
    options: [
      'Queue',
      'Stack (recursion call stack or explicit stack)',
      'Heap',
      'Array',
    ],
    correctAnswer: 1,
    explanation:
      'DFS uses stack (LIFO): recursive DFS uses call stack implicitly, iterative DFS uses explicit stack. Stack ensures deep exploration before backtracking.',
  },
  {
    id: 'mc3',
    question: 'When should you use DFS over BFS?',
    options: [
      'Need shortest path',
      'Exploring all paths, backtracking, topological sort, cycle detection, space matters',
      'Level-order traversal',
      'Always',
    ],
    correctAnswer: 1,
    explanation:
      'Use DFS for: all paths (not just shortest), backtracking problems, topological sort, cycle detection, deep narrow trees (O(H) space better than BFS O(W)). BFS for shortest paths.',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of recursive DFS on a tree?',
    options: [
      'O(N)',
      'O(H) where H is tree height for call stack',
      'O(1)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'Recursive DFS uses O(H) space for call stack where H is height. For balanced tree H=log N, for skewed tree H=N. Plus O(N) for visited set in graphs.',
  },
  {
    id: 'mc5',
    question: 'What are the three tree traversal orders in DFS?',
    options: [
      'Left, right, middle',
      'Preorder (root,left,right), Inorder (left,root,right), Postorder (left,right,root)',
      'Forward, backward, sideways',
      'Random orders',
    ],
    correctAnswer: 1,
    explanation:
      'Three DFS traversals: Preorder (process root first), Inorder (process root between children - BST gives sorted), Postorder (process root last - good for deletion). Order determines when node is processed.',
  },
];
