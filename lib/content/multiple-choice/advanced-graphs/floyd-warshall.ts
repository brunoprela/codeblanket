/**
 * Multiple choice questions for Floyd-Warshall Algorithm section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const floydwarshallMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does Floyd-Warshall compute?',
    options: [
      'Single-source shortest path',
      'All-pairs shortest paths - shortest between every pair of vertices',
      'MST',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Floyd-Warshall: all-pairs shortest paths in O(V³). Dynamic programming approach. Computes dist[i][j] for every pair. Handles negative weights but not negative cycles.',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of Floyd-Warshall?',
    options: [
      'O(E log V)',
      'O(V³) - three nested loops over all vertices',
      'O(VE)',
      'O(V²)',
    ],
    correctAnswer: 1,
    explanation:
      'Floyd-Warshall: O(V³) with three nested loops (k, i, j). For each pair (i,j), try all intermediate vertices k. Space O(V²) for distance matrix.',
  },
  {
    id: 'mc3',
    question: 'How does Floyd-Warshall work?',
    options: [
      'BFS from each vertex',
      'DP: try each vertex k as intermediate, update dist[i][j] = min (dist[i][j], dist[i][k]+dist[k][j])',
      'Greedy',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Floyd-Warshall DP: for each intermediate vertex k, for all pairs (i,j), check if path through k is shorter. dist[i][j] = min (direct, via k). After k loops, all shortest paths found.',
  },
  {
    id: 'mc4',
    question: 'When should you use Floyd-Warshall?',
    options: [
      'Always',
      'Dense graph, need all-pairs, or graph is small (V³ acceptable)',
      'Large sparse graphs',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Use Floyd-Warshall when: 1) Need distances between all pairs, 2) Dense graph (E≈V²), 3) Small V (V³ acceptable), 4) Simple implementation. For sparse: run Dijkstra V times = O(VE log V) faster.',
  },
  {
    id: 'mc5',
    question: 'How do you detect negative cycles in Floyd-Warshall?',
    options: [
      'Cannot detect',
      'After algorithm, if dist[i][i] < 0 for any i, negative cycle exists',
      'Count edges',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'After Floyd-Warshall, check diagonal: if dist[i][i] < 0, vertex i is part of negative cycle. Normal shortest path from vertex to itself is 0. Negative indicates negative cycle.',
  },
];
