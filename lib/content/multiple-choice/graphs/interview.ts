/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a graph problem?',
    options: [
      'Array, list',
      'Network, graph, nodes/edges, dependencies, connections, paths',
      'Sorting',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords: "network", "graph", "nodes/edges", "dependencies", "connections", "paths", "islands", "relationships". These indicate graph structure and traversal.',
  },
  {
    id: 'mc2',
    question: 'What should you clarify first in a graph interview?',
    options: [
      'Complexity only',
      'Directed/undirected? Weighted? Cycles? Connected? Representation?',
      'Language',
      'Nothing',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Directed vs undirected, 2) Weighted edges?, 3) Cycles allowed?, 4) Connected or multiple components?, 5) Representation (adjacency list/matrix). These determine algorithm choice.',
  },
  {
    id: 'mc3',
    question: 'What is a common mistake in graph problems?',
    options: [
      'Using traversal',
      'Forgetting visited set - causes infinite loops in cycles',
      'Good naming',
      'Complexity analysis',
    ],
    correctAnswer: 1,
    explanation:
      'Most common: forgetting visited set. Without it, cycles cause infinite loops. Always track visited nodes to prevent revisiting and ensure O(V+E) complexity.',
  },
  {
    id: 'mc4',
    question: 'When should you choose BFS over DFS in an interview?',
    options: [
      'Always',
      'Shortest path unweighted, level-order, minimum steps',
      'Any problem',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Choose BFS for: shortest path in unweighted graph, level-order traversal, minimum steps/moves. BFS explores by distance, finding closest first.',
  },
  {
    id: 'mc5',
    question: 'What is good practice for graph interview communication?',
    options: [
      'Just code',
      'Clarify, explain approach (BFS/DFS/pattern), walk through example, discuss complexity',
      'Write fast',
      'Skip explanation',
    ],
    correctAnswer: 1,
    explanation:
      'Structure: 1) Clarify graph properties, 2) Identify pattern (traversal, shortest path, connected components), 3) Explain algorithm choice, 4) Walk through example, 5) Complexity analysis.',
  },
];
