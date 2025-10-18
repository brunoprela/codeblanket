/**
 * Multiple choice questions for Time and Space Complexity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of DFS on a tree?',
    options: ['O(log N)', 'O(N) - visit each node once', 'O(N²)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Tree DFS visits each of N nodes exactly once. Each node is processed once. Total O(N) time. Each edge traversed twice (down and back up during recursion).',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of DFS on a graph?',
    options: [
      'O(V)',
      'O(V + E) - visit each vertex once, explore each edge once',
      'O(V²)',
      'O(E)',
    ],
    correctAnswer: 1,
    explanation:
      'Graph DFS: O(V) to visit each vertex with adjacency list, O(E) to explore all edges across all vertices. Total O(V+E). With adjacency matrix, neighbor checking is O(V²).',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of recursive DFS?',
    options: [
      'O(1)',
      'O(H) for call stack where H is depth (H=log N balanced, H=N skewed)',
      'O(N²)',
      'O(E)',
    ],
    correctAnswer: 1,
    explanation:
      'Recursive DFS uses O(H) space for call stack where H is tree/graph depth. Balanced tree: H=log N. Skewed/linear: H=N. Graphs also need O(V) for visited set.',
  },
  {
    id: 'mc4',
    question: 'Why is DFS space-efficient for deep narrow trees?',
    options: [
      'Random',
      'Uses O(H) space proportional to depth, not width - good for narrow deep structures',
      'Always O(1)',
      'No benefit',
    ],
    correctAnswer: 1,
    explanation:
      "DFS uses O(H) space (tree height). For deep narrow trees (H large, W small), this is better than BFS's O(W) width. Example: linked list H=N, W=1: DFS O(N), BFS O(1).",
  },
  {
    id: 'mc5',
    question: 'What affects DFS space complexity more: depth or width?',
    options: [
      'Width',
      'Depth - stack grows with recursion depth O(H)',
      'Both equally',
      'Neither',
    ],
    correctAnswer: 1,
    explanation:
      "DFS space depends on depth (call stack height H), not width. Wide trees don't affect DFS space (siblings don't stack). Deep trees increase stack. Contrast: BFS space depends on width.",
  },
];
