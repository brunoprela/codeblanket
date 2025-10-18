/**
 * Multiple choice questions for Graph Algorithm Selection Guide section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const algorithmselectionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which algorithm should you use for shortest path in a weighted graph with negative edges?',
    options: ["Dijkstra's algorithm", 'BFS', 'Bellman-Ford algorithm', 'DFS'],
    correctAnswer: 2,
    explanation:
      'Bellman-Ford handles negative edge weights correctly. Dijkstra fails with negative weights because it assumes visited nodes have final shortest distances.',
  },
  {
    id: 'mc2',
    question: 'When is BFS preferred over DFS?',
    options: [
      'When you need to detect cycles',
      'When you need shortest path in unweighted graph',
      'When you need topological sort',
      'When memory is very limited',
    ],
    correctAnswer: 1,
    explanation:
      'BFS explores level-by-level and guarantees shortest path in unweighted graphs. DFS is better for cycles, topological sort, and memory constraints.',
  },
  {
    id: 'mc3',
    question: 'What is the main advantage of A* over Dijkstra?',
    options: [
      'A* works with negative weights',
      'A* uses a heuristic to explore fewer nodes toward a known goal',
      'A* is always faster',
      'A* finds all shortest paths',
    ],
    correctAnswer: 1,
    explanation:
      'A* uses a heuristic function to estimate distance to goal, allowing it to prioritize promising paths and explore fewer nodes. It requires knowing the goal location.',
  },
  {
    id: 'mc4',
    question: "Which data structure is essential for Kruskal's MST algorithm?",
    options: [
      'Priority Queue',
      'Stack',
      'Union-Find (Disjoint Set)',
      'Hash Table',
    ],
    correctAnswer: 2,
    explanation:
      "Kruskal's algorithm sorts edges by weight and uses Union-Find to efficiently check if adding an edge creates a cycle.",
  },
  {
    id: 'mc5',
    question: 'What is the time complexity of Dijkstra with a min-heap?',
    options: ['O(V + E)', 'O(V²)', 'O((V + E) log V)', 'O(V · E)'],
    correctAnswer: 2,
    explanation:
      'With a min-heap (priority queue), Dijkstra is O((V + E) log V): each vertex is extracted once (V log V) and each edge relaxation may update the heap (E log V).',
  },
];
