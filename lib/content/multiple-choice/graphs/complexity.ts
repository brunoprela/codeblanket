/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of BFS/DFS traversal?',
    options: [
      'O(V)',
      'O(V + E) - visit all vertices and edges once',
      'O(E)',
      'O(V²)',
    ],
    correctAnswer: 1,
    explanation:
      'BFS/DFS: O(V + E). Visit each vertex once (O(V)), explore each edge once (O(E)). Total: O(V + E). Space: O(V) for visited set.',
  },
  {
    id: 'mc2',
    question:
      "What is the time complexity of Dijkstra\'s algorithm with min heap?",
    options: [
      'O(V + E)',
      'O((V + E) log V) - heap operations for all edges',
      'O(V²)',
      'O(E log E)',
    ],
    correctAnswer: 1,
    explanation:
      'Dijkstra with min heap: each edge causes heap operation O(log V). Process O(V + E) edges total. Time: O((V + E) log V). Better than O(V²) for sparse graphs.',
  },
  {
    id: 'mc3',
    question: 'When do you need Bellman-Ford instead of Dijkstra?',
    options: [
      'Always',
      'When graph has negative weight edges',
      'For faster execution',
      'Random choice',
    ],
    correctAnswer: 1,
    explanation:
      'Dijkstra fails with negative weights. Bellman-Ford handles negative edges and detects negative cycles. Time: O(V*E) vs Dijkstra O((V+E) log V).',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of graph traversals?',
    options: ['O(1)', 'O(V) for visited set and queue/stack', 'O(E)', 'O(V²)'],
    correctAnswer: 1,
    explanation:
      'Graph traversals: O(V) space for visited set + queue/stack can hold up to O(V) nodes. DFS recursion: O(H) where H is depth.',
  },
  {
    id: 'mc5',
    question:
      'Why is adjacency list better than adjacency matrix for sparse graphs?',
    options: [
      'Faster edge lookup',
      'Space: O(V+E) vs O(V²), and most real graphs are sparse',
      'Random',
      'Always better',
    ],
    correctAnswer: 1,
    explanation:
      'Sparse graphs have few edges (E << V²). List uses O(V+E) space, matrix O(V²). Social networks, web graphs are sparse, making list much more efficient.',
  },
];
