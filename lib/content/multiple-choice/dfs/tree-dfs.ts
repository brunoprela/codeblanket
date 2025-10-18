/**
 * Multiple choice questions for DFS on Trees section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const treedfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is Preorder traversal and when do you use it?',
    options: [
      'Left, root, right',
      'Root, left, right - use for copying tree, serialization, prefix expression',
      'Left, right, root',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Preorder: process root first, then left subtree, then right. Use when you need to process parent before children: tree copying, serialization, prefix notation evaluation.',
  },
  {
    id: 'mc2',
    question: 'Why is Inorder traversal special for BST?',
    options: [
      'Random',
      'Visits nodes in sorted order - validates BST property',
      'Fastest',
      'Uses least space',
    ],
    correctAnswer: 1,
    explanation:
      'Inorder (left, root, right) on BST visits nodes in ascending sorted order because of BST property (left < root < right). Perfect for BST validation and sorted output.',
  },
  {
    id: 'mc3',
    question: 'When should you use Postorder traversal?',
    options: [
      'Always',
      'Process children before parent - tree deletion, computing height, postfix expressions',
      'Never',
      'BST only',
    ],
    correctAnswer: 1,
    explanation:
      'Postorder: process children first, then parent. Use when parent depends on children: deleting tree (delete children first), computing height (need children heights), postfix evaluation.',
  },
  {
    id: 'mc4',
    question: 'What is Morris Traversal?',
    options: [
      'BFS variant',
      'Inorder traversal in O(1) space using threaded binary tree (temporary links)',
      'Sorting algorithm',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Morris Traversal achieves inorder traversal in O(1) space by temporarily modifying tree (creating threads to successors). Avoids recursion/stack. Restores tree after traversal.',
  },
  {
    id: 'mc5',
    question: 'How do you implement iterative DFS traversal?',
    options: [
      'Use queue',
      'Use explicit stack - push right child first, then left (so left pops first)',
      'Impossible',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Iterative DFS uses stack: push root, loop (pop node, process, push right child, push left child). Push right first so left pops first (LIFO), maintaining left-to-right order.',
  },
];
