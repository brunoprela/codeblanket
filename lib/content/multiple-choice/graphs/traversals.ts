/**
 * Multiple choice questions for Graph Traversals: BFS and DFS section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const traversalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the key difference between BFS and DFS?',
    options: [
      'Speed',
      'BFS explores level-by-level (queue), DFS explores depth-first (stack/recursion)',
      'Space only',
      'They are the same',
    ],
    correctAnswer: 1,
    explanation:
      'BFS uses queue to explore level-by-level (closest first). DFS uses stack/recursion to explore depth-first (go deep before backtracking). Different exploration orders.',
  },
  {
    id: 'mc2',
    question: 'When should you use BFS over DFS?',
    options: [
      'Always',
      'Shortest path unweighted, level-order, closest nodes first',
      'Any traversal',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Use BFS for: shortest path in unweighted graph (finds closest), level-order traversal, minimum moves. BFS explores by distance from start.',
  },
  {
    id: 'mc3',
    question: 'When should you use DFS over BFS?',
    options: [
      'Never',
      'Pathfinding, cycle detection, topological sort, connected components',
      'Shortest path only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Use DFS for: any path (not shortest), cycle detection, topological sort, connected components, backtracking. DFS explores deeply, better for path existence.',
  },
  {
    id: 'mc4',
    question: 'What data structure does BFS use?',
    options: [
      'Stack',
      'Queue (FIFO) - explores level-by-level',
      'Heap',
      'Array',
    ],
    correctAnswer: 1,
    explanation:
      'BFS uses queue (FIFO). Add neighbors to queue, process in order added. This ensures level-by-level exploration (all distance k before k+1).',
  },
  {
    id: 'mc5',
    question: 'Why track visited nodes in graph traversals?',
    options: [
      'For speed',
      'Prevents infinite loops in cycles, ensures O(V+E) time',
      'Random requirement',
      'Memory optimization',
    ],
    correctAnswer: 1,
    explanation:
      'Visited set prevents revisiting nodes. Without it, cycles cause infinite loops. With it, each node processed once, giving O(V+E) time complexity.',
  },
];
