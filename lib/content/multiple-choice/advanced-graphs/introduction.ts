/**
 * Multiple choice questions for Advanced Graph Algorithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What distinguishes advanced graph algorithms from basic traversals?',
    options: [
      'Just faster',
      'Solve optimization problems (shortest path, MST, max flow) vs simple exploration',
      'Random',
      'Use more memory',
    ],
    correctAnswer: 1,
    explanation:
      'Advanced algorithms solve optimization: shortest paths (Dijkstra, Bellman-Ford), MST (Prim, Kruskal), max flow, SCC. Basic (BFS/DFS) just explore/traverse. Different goals and complexities.',
  },
  {
    id: 'mc2',
    question: 'When should you use Dijkstra vs Bellman-Ford?',
    options: [
      'Always Dijkstra',
      'Dijkstra: non-negative weights O(E log V). Bellman-Ford: negative weights allowed O(VE)',
      'Random',
      'Same algorithm',
    ],
    correctAnswer: 1,
    explanation:
      'Dijkstra faster O(E log V) but requires non-negative weights. Bellman-Ford slower O(VE) but handles negative weights and detects negative cycles. BFS for unweighted.',
  },
  {
    id: 'mc3',
    question: 'What is a Minimum Spanning Tree?',
    options: [
      'Shortest path',
      'Tree connecting all vertices with minimum total edge weight - no cycles',
      'Largest tree',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'MST: tree (no cycles) spanning all vertices with minimum sum of edge weights. Used in network design, clustering. Algorithms: Prim O(E log V), Kruskal O(E log E).',
  },
  {
    id: 'mc4',
    question: 'What problem does Floyd-Warshall solve?',
    options: [
      'Single-source shortest path',
      'All-pairs shortest paths - shortest between every pair of vertices',
      'MST',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Floyd-Warshall: all-pairs shortest paths in O(V³). Computes shortest path between every pair. Good for dense graphs or when need all distances. Handles negative weights.',
  },
  {
    id: 'mc5',
    question: 'What is network flow used for?',
    options: [
      'Traversal',
      'Maximum flow through network (capacity constraints) - resource allocation, matching',
      'Sorting',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Network flow: maximum flow from source to sink respecting edge capacities. Applications: resource allocation, bipartite matching, assignment problems. Algorithms: Ford-Fulkerson, Edmonds-Karp.',
  },
];
