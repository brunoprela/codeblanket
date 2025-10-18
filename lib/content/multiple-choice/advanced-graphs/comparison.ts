/**
 * Multiple choice questions for Algorithm Comparison section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const comparisonMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do shortest path algorithms compare?',
    options: [
      'All same',
      'BFS: unweighted O(V+E). Dijkstra: non-negative O(E log V). Bellman-Ford: negative O(VE). Floyd-Warshall: all-pairs O(V³)',
      'Random',
      'All O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'BFS fastest for unweighted. Dijkstra fastest for non-negative weights. Bellman-Ford handles negatives. Floyd-Warshall for all-pairs. Choose based on graph properties and requirements.',
  },
  {
    id: 'mc2',
    question: 'When should you use BFS vs Dijkstra?',
    options: [
      'Always Dijkstra',
      'BFS: unweighted graphs O(V+E). Dijkstra: weighted non-negative O(E log V)',
      'Random',
      'Same thing',
    ],
    correctAnswer: 1,
    explanation:
      'BFS is special case of Dijkstra for unweighted (all weights = 1). BFS O(V+E) simpler and faster. Dijkstra O(E log V) generalizes to weighted. Use simplest algorithm that works.',
  },
  {
    id: 'mc3',
    question: 'How do MST algorithms compare?',
    options: [
      'Same algorithm',
      'Kruskal: edge-based O(E log E), good for sparse. Prim: vertex-based O(E log V), good for dense',
      'Random',
      'Both O(V³)',
    ],
    correctAnswer: 1,
    explanation:
      'Kruskal: sort edges, add if no cycle. O(E log E). Prim: grow tree, add min edge. O(E log V). Kruskal better for sparse (few edges), Prim for dense (many edges).',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity trade-off?',
    options: [
      'All same',
      'BFS/Dijkstra/Bellman-Ford: O(V). Floyd-Warshall: O(V²) for all-pairs matrix',
      'All O(1)',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Single-source algorithms (BFS, Dijkstra, Bellman-Ford): O(V) for distances. All-pairs (Floyd-Warshall): O(V²) matrix. Trade-off: space vs complete information.',
  },
  {
    id: 'mc5',
    question: 'How do you choose the right algorithm?',
    options: [
      'Random',
      'Consider: weighted? negative? all-pairs? sparse/dense? Then match to algorithm constraints and complexity',
      'Always use Dijkstra',
      'No method',
    ],
    correctAnswer: 1,
    explanation:
      'Decision tree: 1) Unweighted → BFS, 2) Non-negative weighted → Dijkstra, 3) Negative weights → Bellman-Ford, 4) All-pairs → Floyd-Warshall, 5) MST → Kruskal (sparse) or Prim (dense).',
  },
];
