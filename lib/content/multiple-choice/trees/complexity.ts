/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of search in a balanced BST?',
    options: ['O(N)', 'O(log N)', 'O(1)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'In a balanced BST with height log N, each comparison eliminates half the remaining nodes, giving O(log N) time complexity for search, insert, and delete operations.',
  },
  {
    id: 'mc2',
    question:
      'What happens to BST operation complexity in a skewed (unbalanced) tree?',
    options: [
      'Stays O(log N)',
      'Degrades to O(N)',
      'Becomes O(1)',
      'Becomes O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'In a skewed tree, height can be N (like a linked list). Operations must traverse from root to leaf, taking O(N) time. This is why balance is crucial for BST efficiency.',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of recursive DFS?',
    options: [
      'O(1)',
      'O(H) where H is tree height',
      'O(N)',
      'O(W) where W is width',
    ],
    correctAnswer: 1,
    explanation:
      'Recursive DFS uses O(H) space for the call stack where H is height. The stack stores nodes along one path from root to current node. In balanced trees H = log N, in skewed trees H = N.',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of BFS (level-order traversal)?',
    options: [
      'O(H) where H is height',
      'O(W) where W is maximum width',
      'O(1)',
      'O(log N)',
    ],
    correctAnswer: 1,
    explanation:
      'BFS uses a queue that stores all nodes at the current level. Maximum queue size is the width W of the tree. In a complete binary tree, this can be N/2 nodes at the bottom level.',
  },
  {
    id: 'mc5',
    question: 'Why are all basic tree traversals O(N) time complexity?',
    options: [
      'They use nested loops',
      'They visit each of N nodes exactly once, doing constant work per node',
      'They are slow',
      'They require sorting',
    ],
    correctAnswer: 1,
    explanation:
      'All traversals (inorder, preorder, postorder, level-order) visit each node exactly once and perform constant work at each node. Total work is N × O(1) = O(N).',
  },
];
