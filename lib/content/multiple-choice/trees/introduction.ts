/**
 * Multiple choice questions for Introduction to Trees section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What makes trees hierarchical structures?',
    options: [
      'They are stored in memory sequentially',
      'Each node can have multiple children, creating parent-child relationships in levels',
      'They are always sorted',
      'They use arrays internally',
    ],
    correctAnswer: 1,
    explanation:
      'Trees are hierarchical because nodes can have multiple children, creating parent-child relationships across levels. This branching structure differs from linear structures like arrays or linked lists.',
  },
  {
    id: 'mc2',
    question:
      'What is the maximum number of children a node can have in a binary tree?',
    options: ['Unlimited', '2', '3', '1'],
    correctAnswer: 1,
    explanation:
      'Binary trees limit each node to at most 2 children: left and right. This constraint enables specific algorithms like BST with efficient O(log N) search.',
  },
  {
    id: 'mc3',
    question: 'What property defines a Binary Search Tree?',
    options: [
      'All nodes have exactly 2 children',
      'Left subtree < node < right subtree for all nodes',
      'All leaves are at the same level',
      'Height is always balanced',
    ],
    correctAnswer: 1,
    explanation:
      'BST property: for every node, all values in left subtree are less than node value, and all values in right subtree are greater. This ordering enables O(log N) search in balanced trees.',
  },
  {
    id: 'mc4',
    question: 'What is a leaf node?',
    options: [
      'The root node',
      'A node with no children',
      'A node with one child',
      'A node with two children',
    ],
    correctAnswer: 1,
    explanation:
      'A leaf node is a node with no children - it is at the bottom of the tree. Leaves are terminal nodes in tree traversals.',
  },
  {
    id: 'mc5',
    question:
      'In a balanced BST with 1000 nodes, approximately how many comparisons are needed to search for a value?',
    options: ['1000', '10', '100', '500'],
    correctAnswer: 1,
    explanation:
      'A balanced tree with 1000 nodes has height log₂(1000) ≈ 10. Search makes one comparison per level, so at most 10 comparisons are needed (since 2^10 = 1024).',
  },
];
