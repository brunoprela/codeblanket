/**
 * Multiple choice questions for Union-Find (Disjoint Set Union) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const unionfindMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is Union-Find used for?',
    options: [
      'Sorting',
      'Disjoint set operations - track connected components, detect cycles',
      'Shortest path',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Union-Find (Disjoint Set Union): efficiently track which vertices are in same connected component. Operations: find(x) finds root, union(x,y) merges sets. Used in Kruskal's MST, cycle detection.",
  },
  {
    id: 'mc2',
    question: 'What optimizations make Union-Find efficient?',
    options: [
      'None',
      'Path compression + union by rank - nearly O(1) amortized per operation',
      'Sorting',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Optimizations: 1) Path compression: during find, point all nodes to root (flattens tree), 2) Union by rank: attach smaller tree to larger (keeps tree shallow). Together: O(α(N)) ≈ O(1) amortized.',
  },
  {
    id: 'mc3',
    question: 'How does path compression work?',
    options: [
      'Deletes paths',
      'During find(x), make all nodes on path point directly to root',
      'Random',
      'Sorts nodes',
    ],
    correctAnswer: 1,
    explanation:
      'Path compression: when finding root of x, update parent of all nodes on path to point directly to root. Flattens tree structure. Future finds on same path become O(1).',
  },
  {
    id: 'mc4',
    question: 'What is union by rank?',
    options: [
      'Random union',
      'Always attach shorter tree under taller tree to keep balanced',
      'Sort first',
      'Union by size',
    ],
    correctAnswer: 1,
    explanation:
      'Union by rank: track tree height (rank). When merging, make root of shorter tree point to root of taller. Keeps tree shallow. Alternative: union by size (attach smaller to larger).',
  },
  {
    id: 'mc5',
    question:
      'What is the amortized time complexity of Union-Find with optimizations?',
    options: [
      'O(log N)',
      'O(α(N)) where α is inverse Ackermann - effectively O(1)',
      'O(N)',
      'O(1) exact',
    ],
    correctAnswer: 1,
    explanation:
      'With path compression + union by rank: O(α(N)) amortized per operation, where α is inverse Ackermann function. α(N) ≤ 5 for any practical N. Effectively constant time.',
  },
];
