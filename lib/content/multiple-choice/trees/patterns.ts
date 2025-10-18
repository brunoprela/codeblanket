/**
 * Multiple choice questions for Common Tree Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What are the three essential steps in the recursive DFS pattern?',
    options: [
      'Initialize, loop, return',
      'Base case, recurse on children, compute current result',
      'Sort, search, merge',
      'Read, write, update',
    ],
    correctAnswer: 1,
    explanation:
      'The recursive DFS pattern consists of: 1) Base case (null check, return default), 2) Recurse on children (get results from subtrees), 3) Compute current result (combine children results with current node).',
  },
  {
    id: 'mc2',
    question: 'Why is passing bounds necessary when validating a BST?',
    options: [
      'To make it faster',
      'To ensure nodes satisfy BST property relative to all ancestors, not just parent',
      'To save memory',
      'It is not necessary',
    ],
    correctAnswer: 1,
    explanation:
      'Without bounds, checking only node > left and node < right misses violations deeper in the tree. Bounds ensure each node satisfies BST property relative to all ancestors, maintaining the global BST property.',
  },
  {
    id: 'mc3',
    question: 'What is backtracking in the path pattern and why is it crucial?',
    options: [
      'Going backwards in the tree',
      'Removing nodes from the path after recursion to maintain correct state',
      'Deleting nodes',
      'Reversing the tree',
    ],
    correctAnswer: 1,
    explanation:
      'Backtracking removes the current node from the path after exploring its subtree. This ensures the path is correct when exploring the other subtree. Without backtracking, the path would incorrectly contain nodes from both subtrees.',
  },
  {
    id: 'mc4',
    question:
      'What is the difference between top-down and bottom-up tree traversal?',
    options: [
      'Top-down goes left, bottom-up goes right',
      'Top-down passes info from parent to children, bottom-up gathers info from children to parent',
      'They are the same',
      'Top-down is faster',
    ],
    correctAnswer: 1,
    explanation:
      'Top-down (preorder-style) passes information down from parent to children (e.g., tracking depth). Bottom-up (postorder-style) gathers information from children to compute parent result (e.g., calculating height).',
  },
  {
    id: 'mc5',
    question:
      'When comparing two trees for structure and values, what is the base case?',
    options: [
      'Both are leaves',
      'Both are null (return true) or one is null (return false)',
      'Values are equal',
      'Trees are balanced',
    ],
    correctAnswer: 1,
    explanation:
      'Base case: if both trees are null, they are equal (return true). If only one is null, they differ in structure (return false). Then check if values match and recurse on subtrees.',
  },
];
