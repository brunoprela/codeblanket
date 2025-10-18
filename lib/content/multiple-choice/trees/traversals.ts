/**
 * Multiple choice questions for Tree Traversals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const traversalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What order does inorder traversal visit nodes in a BST?',
    options: [
      'Random order',
      'Ascending sorted order',
      'Descending order',
      'Level order',
    ],
    correctAnswer: 1,
    explanation:
      'Inorder traversal (left, node, right) visits nodes in ascending sorted order for a Binary Search Tree. This is because it processes left (smaller values) before node before right (larger values).',
  },
  {
    id: 'mc2',
    question: 'Which traversal processes the parent node before its children?',
    options: ['Inorder', 'Preorder', 'Postorder', 'Level order'],
    correctAnswer: 1,
    explanation:
      'Preorder traversal (node, left, right) processes the parent before its children, making it useful for copying trees or creating prefix expressions where parent context is needed first.',
  },
  {
    id: 'mc3',
    question: 'What data structure does BFS use for tree traversal?',
    options: ['Stack', 'Queue', 'Array', 'Hash map'],
    correctAnswer: 1,
    explanation:
      'BFS uses a queue to explore nodes level by level. Nodes are added to the queue and processed in FIFO order, ensuring each level is completed before moving to the next.',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of recursive DFS?',
    options: ['O(1)', 'O(H) where H is tree height', 'O(N)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'Recursive DFS uses O(H) space for the call stack where H is tree height. In the worst case (skewed tree), H = N. In a balanced tree, H = log N.',
  },
  {
    id: 'mc5',
    question: 'When is BFS preferred over DFS?',
    options: [
      'When solutions are deep in the tree',
      'When finding shortest path or solutions near the root',
      'When space is limited',
      'Always',
    ],
    correctAnswer: 1,
    explanation:
      'BFS is preferred for finding shortest paths (minimum depth to leaf) or when solutions are likely near the root. BFS guarantees finding nodes level by level in order of distance from root.',
  },
];
