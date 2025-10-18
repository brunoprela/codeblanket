/**
 * Multiple choice questions for Bellman-Ford Algorithm section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bellmanfordMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does Bellman-Ford handle that Dijkstra cannot?',
    options: [
      'Large graphs',
      'Negative edge weights and negative cycle detection',
      'Directed graphs',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Bellman-Ford handles negative weights and detects negative cycles. Relaxes all edges V-1 times. Slower O(VE) than Dijkstra O(E log V), but more versatile.',
  },
  {
    id: 'mc2',
    question: 'How does Bellman-Ford detect negative cycles?',
    options: [
      'Cannot detect',
      'After V-1 iterations, if any edge can still relax, negative cycle exists',
      'Count edges',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Shortest path has at most V-1 edges. After V-1 iterations, distances should be final. If Vth iteration still improves distance, cycle with negative weight exists.',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of Bellman-Ford?',
    options: [
      'O(E log V)',
      'O(VE) - V-1 iterations, each relaxes E edges',
      'O(V²)',
      'O(V³)',
    ],
    correctAnswer: 1,
    explanation:
      'Bellman-Ford: V-1 iterations of relaxing all E edges = O(VE). Slower than Dijkstra O(E log V) but handles negative weights. Dense graph: O(V³).',
  },
  {
    id: 'mc4',
    question: 'Why does Bellman-Ford require V-1 iterations?',
    options: [
      'Random',
      'Shortest simple path has at most V-1 edges - each iteration extends path by 1 edge',
      'Optimization',
      'Historical',
    ],
    correctAnswer: 1,
    explanation:
      'Shortest simple path (no repeated vertices) has at most V-1 edges. Iteration i finds shortest paths with ≤i edges. After V-1 iterations, all shortest paths found.',
  },
  {
    id: 'mc5',
    question: 'When should you use Bellman-Ford?',
    options: [
      'Always',
      'When graph has negative weights, need negative cycle detection, or distributed/simple implementation',
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use Bellman-Ford when: 1) Negative weights present, 2) Need to detect negative cycles, 3) Distributed systems (simple to parallelize), 4) Small graphs where O(VE) acceptable. Otherwise Dijkstra faster.',
  },
];
