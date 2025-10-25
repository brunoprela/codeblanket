/**
 * Multiple choice questions for Dijkstra section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dijkstraMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: "What is Dijkstra\'s algorithm used for?",
    options: [
      'MST',
      'Single-source shortest path with non-negative weights',
      'All-pairs shortest path',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Dijkstra finds shortest path from one source to all vertices. Requires non-negative edge weights. Uses priority queue (min-heap). O(E log V) time.',
  },
  {
    id: 'mc2',
    question: 'Why does Dijkstra use a priority queue?',
    options: [
      'Random choice',
      'Always process closest unvisited vertex next - greedy choice ensures optimality',
      'Faster traversal',
      'Required',
    ],
    correctAnswer: 1,
    explanation:
      'Dijkstra greedy: always process vertex with minimum distance. Priority queue extracts min in O(log V). Ensures when vertex processed, shortest path found. Without queue: O(V²).',
  },
  {
    id: 'mc3',
    question: 'What is the time complexity of Dijkstra?',
    options: [
      'O(V)',
      'O(E log V) with min-heap, O(V²) with array',
      'O(V³)',
      'O(E)',
    ],
    correctAnswer: 1,
    explanation:
      'With min-heap: V extractions O(V log V) + E updates O(E log V) = O(E log V). With array: find min O(V) for V vertices = O(V²). Dense graphs: array better.',
  },
  {
    id: 'mc4',
    question: 'Why does Dijkstra fail with negative weights?',
    options: [
      'Random',
      'Greedy assumes found path is shortest - negative edges can improve later, violating assumption',
      'Too slow',
      'No reason',
    ],
    correctAnswer: 1,
    explanation:
      'Dijkstra greedy: once vertex processed, distance is final. Negative edges can make later path shorter, breaking assumption. Example: A→B=5, A→C=2, C→B=-10 gives A→C→B=-8 < 5.',
  },
  {
    id: 'mc5',
    question: 'How do you reconstruct the shortest path in Dijkstra?',
    options: [
      'Cannot reconstruct',
      'Track parent/previous vertex for each node, backtrack from target to source',
      'Store all paths',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'During Dijkstra, when updating dist[v], store parent[v] = u. After algorithm, backtrack: path = [target], while path[-1] != source: path.append (parent[path[-1]]). Reverse for source→target.',
  },
];
