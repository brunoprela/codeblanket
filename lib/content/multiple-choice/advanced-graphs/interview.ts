/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal an advanced graph problem?',
    options: [
      'Traversal',
      'Shortest path, minimum/maximum, optimal, flow, spanning tree, connectivity',
      'Random',
      'Simple graph',
    ],
    correctAnswer: 1,
    explanation:
      'Advanced graph keywords: "shortest path", "minimum cost", "spanning tree", "maximum flow", "strongly connected", "bridges/articulation". Suggests optimization algorithms beyond BFS/DFS.',
  },
  {
    id: 'mc2',
    question: 'How do you approach a shortest path interview question?',
    options: [
      'Always Dijkstra',
      'Clarify: single-source or all-pairs? weights? negative? Then choose: BFS/Dijkstra/Bellman-Ford/Floyd-Warshall',
      'Random',
      'BFS only',
    ],
    correctAnswer: 1,
    explanation:
      'Approach: 1) Clarify single-source vs all-pairs, 2) Check weights (none/positive/negative), 3) Choose algorithm: unweighted→BFS, non-negative→Dijkstra, negative→Bellman-Ford, all-pairs→Floyd-Warshall.',
  },
  {
    id: 'mc3',
    question: 'What should you clarify in a graph interview?',
    options: [
      'Nothing',
      'Directed/undirected? Weighted? Connected? Constraints on V,E? Dense/sparse?',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Directed or undirected, 2) Weighted (and range), 3) Connected or multiple components, 4) V,E size (affects algorithm choice), 5) Dense (E≈V²) or sparse (E≈V).',
  },
  {
    id: 'mc4',
    question: 'What is a common graph algorithm mistake?',
    options: [
      'Using graphs',
      'Wrong algorithm (Dijkstra with negative weights), not handling disconnected components, off-by-one in adjacency',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Dijkstra with negative weights (use Bellman-Ford), 2) Forgetting disconnected components (loop all vertices), 3) Wrong complexity analysis, 4) Not initializing distances correctly.',
  },
  {
    id: 'mc5',
    question: 'How should you communicate your graph solution?',
    options: [
      'Just code',
      'Explain algorithm choice (why Dijkstra vs BFS?), complexity, walk through example, discuss trade-offs',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Why this algorithm (graph properties→algorithm), 2) How it works briefly, 3) Walk through small example, 4) Time O(?) and space O(?), 5) Trade-offs (Kruskal vs Prim).',
  },
];
