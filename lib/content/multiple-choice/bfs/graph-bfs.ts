/**
 * Multiple choice questions for BFS on Graphs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const graphbfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Why do you need a visited set for graph BFS?',
    options: [
      'Optimization',
      'Prevent infinite loops from cycles - mark nodes as visited',
      'Random requirement',
      'Faster',
    ],
    correctAnswer: 1,
    explanation:
      'Graphs can have cycles. Without visited set, BFS would enqueue same nodes repeatedly, causing infinite loops. Visited set ensures each node processed once.',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of BFS on a graph?',
    options: [
      'O(V)',
      'O(V + E) - visit each vertex once, explore each edge once',
      'O(V²)',
      'O(E)',
    ],
    correctAnswer: 1,
    explanation:
      'BFS visits each vertex once O(V), explores each edge once O(E). With adjacency list, checking neighbors sums to O(E) across all vertices. Total: O(V + E).',
  },
  {
    id: 'mc3',
    question: 'How does BFS find shortest path in unweighted graph?',
    options: [
      'Random',
      'First time reaching node is via shortest path - distance tracking with queue',
      'Try all paths',
      'Sorting',
    ],
    correctAnswer: 1,
    explanation:
      'BFS processes nodes by distance. First discovery of node is via shortest path (all shorter paths already explored). Track distance: start at 0, increment for each level.',
  },
  {
    id: 'mc4',
    question: 'How do you find connected components using BFS?',
    options: [
      'Cannot do',
      'For each unvisited node: run BFS (marks component), increment count',
      'DFS only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Iterate through all nodes. For each unvisited: run BFS (marks entire component as visited), increment component count. Each BFS explores one complete component.',
  },
  {
    id: 'mc5',
    question: 'What is 0-1 BFS and when do you use it?',
    options: [
      'Random algorithm',
      'Shortest path in graph with edge weights 0 or 1 - use deque, 0-weight edges go front, 1-weight go back',
      'Normal BFS',
      'Cannot do',
    ],
    correctAnswer: 1,
    explanation:
      "0-1 BFS handles graphs with only 0 or 1 edge weights. Use deque: add 0-weight edges to front (priority), 1-weight to back. Achieves O(V+E) instead of Dijkstra's O(E log V).",
  },
];
