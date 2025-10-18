/**
 * Multiple choice questions for Lowest Common Ancestor (LCA) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const lowestcommonancestorMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of finding LCA in a Binary Search Tree using the iterative approach?',
    options: ['O(N)', 'O(H) where H is height', 'O(log N always)', 'O(1)'],
    correctAnswer: 1,
    explanation:
      'LCA in BST using the iterative approach takes O(H) time where H is the height of the tree. We traverse down from root to the split point. In balanced BST, H = log N; in skewed tree, H = N.',
  },
  {
    id: 'mc2',
    question:
      'For general binary tree LCA, what does it mean when both left and right recursive calls return non-null?',
    options: [
      'There is an error',
      'Current root is the LCA (split point)',
      'Both nodes are in left subtree',
      'Continue searching',
    ],
    correctAnswer: 1,
    explanation:
      'When both left and right return non-null, it means one target node was found in the left subtree and the other in the right subtree. The current root is therefore the lowest common ancestor - the split point where paths to the two nodes diverge.',
  },
  {
    id: 'mc3',
    question:
      'What is the space complexity of the recursive LCA algorithm for binary trees?',
    options: ['O(1)', 'O(H) where H is height', 'O(N)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'The recursive LCA uses O(H) space for the call stack where H is the tree height. Each recursive call adds a frame to the stack, and the maximum depth is the height of the tree.',
  },
  {
    id: 'mc4',
    question:
      'Why can BST LCA be solved more efficiently than general binary tree LCA?',
    options: [
      'BST is always balanced',
      'BST ordering property tells us which subtree to search without exploring both',
      'BST has fewer nodes',
      'BST nodes have parent pointers',
    ],
    correctAnswer: 1,
    explanation:
      'BST LCA is more efficient because the ordering property (left < root < right) tells us exactly which direction to search. If both nodes are less than root, go left; if both are greater, go right. We never need to explore both subtrees, enabling O(1) space iterative solution.',
  },
  {
    id: 'mc5',
    question: 'In LCA problems, what should you clarify about the input?',
    options: [
      'The programming language to use',
      'Whether it is a BST or general binary tree, and if both nodes are guaranteed to exist',
      'The number of nodes in the tree',
      'Whether to use recursion',
    ],
    correctAnswer: 1,
    explanation:
      'Always clarify: 1) BST or general binary tree (different algorithms), 2) Are both nodes guaranteed to exist (affects validation), 3) Can node be its own ancestor (affects base case). These factors determine the approach and edge cases.',
  },
];
