/**
 * Multiple choice questions for Minimum Spanning Tree (MST) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const mstMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is a Minimum Spanning Tree?',
    options: [
      'Shortest path tree',
      'Tree connecting all vertices with minimum total edge weight',
      'Largest tree',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'MST: subset of edges forming tree (connected, acyclic) that includes all vertices with minimum sum of edge weights. Unique for distinct weights. Used in network design.',
  },
  {
    id: 'mc2',
    question: "How does Kruskal's algorithm work?",
    options: [
      'BFS',
      "Sort edges by weight, greedily add edge if doesn't create cycle (use Union-Find)",
      'Random',
      'DFS',
    ],
    correctAnswer: 1,
    explanation:
      'Kruskal: 1) Sort all edges by weight O(E log E), 2) For each edge, if vertices in different components (Union-Find), add edge and union. O(E log E) for sort + O(E α(V)) for unions.',
  },
  {
    id: 'mc3',
    question: "How does Prim's algorithm work?",
    options: [
      'Sorting edges',
      'Start from vertex, grow tree by adding minimum weight edge connecting tree to new vertex (use priority queue)',
      'Random',
      'Union-Find',
    ],
    correctAnswer: 1,
    explanation:
      'Prim: 1) Start with any vertex, 2) Repeatedly add minimum weight edge connecting tree to new vertex (use min-heap), 3) Continue until all vertices included. O(E log V) with heap.',
  },
  {
    id: 'mc4',
    question: 'When should you use Kruskal vs Prim?',
    options: [
      'Same',
      'Kruskal: sparse graphs (edge-focused). Prim: dense graphs (vertex-focused)',
      'Random',
      'Always Prim',
    ],
    correctAnswer: 1,
    explanation:
      'Kruskal O(E log E): better for sparse graphs (fewer edges to sort). Uses Union-Find. Prim O(E log V): better for dense graphs (many edges per vertex). Uses priority queue. Both correct.',
  },
  {
    id: 'mc5',
    question: 'What applications use MST?',
    options: [
      'None',
      'Network design (min cable), clustering, approximation algorithms (TSP)',
      'Only theoretical',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'MST applications: 1) Network design (minimize cable/pipe length), 2) Clustering (cut MST edges), 3) TSP approximation (MST gives lower bound), 4) Image segmentation, 5) Handwriting recognition.',
  },
];
