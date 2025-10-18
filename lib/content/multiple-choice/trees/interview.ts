/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What keywords in a problem description indicate a tree algorithm is needed?',
    options: [
      'Array, sort, search',
      'Binary tree, BST, parent-child, root-to-leaf, inorder',
      'Hash map, frequency, count',
      'String, substring, pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "binary tree", "BST", hierarchical relationships ("parent-child", "ancestor"), traversals ("inorder", "preorder"), and structural properties ("balanced", "symmetric", "depth") indicate tree problems.',
  },
  {
    id: 'mc2',
    question: 'What should you clarify first in a tree interview problem?',
    options: [
      'The test cases',
      "Whether it's a binary tree or BST, and what to return",
      'The programming language',
      'How many nodes',
    ],
    correctAnswer: 1,
    explanation:
      "Always clarify whether it's a binary tree (any values) or BST (ordered values), if there are null nodes, and what the function should return. BST problems can use ordering for optimization.",
  },
  {
    id: 'mc3',
    question: 'What is the most common mistake in tree problems?',
    options: [
      'Using wrong variable names',
      'Forgetting null checks, causing null pointer errors',
      'Using too much memory',
      'Making it too fast',
    ],
    correctAnswer: 1,
    explanation:
      'Forgetting to check if a node is null before accessing its properties causes null pointer errors. Always start recursive functions with "if not root: return default_value".',
  },
  {
    id: 'mc4',
    question:
      'When explaining tree solution complexity, what should you mention?',
    options: [
      'Only time complexity',
      'Time O(N) to visit all nodes, Space O(H) for recursion stack where H is height',
      'Only space complexity',
      'That trees are slow',
    ],
    correctAnswer: 1,
    explanation:
      'Mention time complexity O(N) for visiting all nodes and space complexity O(H) for the recursion stack, where H ranges from log N (balanced) to N (skewed tree).',
  },
  {
    id: 'mc5',
    question: 'What is a recommended practice progression for tree mastery?',
    options: [
      'Start with the hardest problems',
      'Start with basics (traversals, depth), then BST, then paths, then advanced',
      'Only practice BST problems',
      'Skip practice and memorize solutions',
    ],
    correctAnswer: 1,
    explanation:
      'Progress from basic traversals and depth problems, to BST operations, to path problems, and finally advanced topics like serialization and LCA. This builds intuition incrementally.',
  },
];
