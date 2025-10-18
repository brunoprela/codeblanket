/**
 * Multiple choice questions for Introduction to BFS section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the core concept of BFS?',
    options: [
      'Go deep first',
      'Explore level by level, visiting all neighbors before moving to next level',
      'Random traversal',
      'Find all paths',
    ],
    correctAnswer: 1,
    explanation:
      'BFS explores breadth-first (wide) - visits all nodes at current level before moving to next level. Uses queue (FIFO) to process nodes in order of distance from start.',
  },
  {
    id: 'mc2',
    question: 'What data structure does BFS use?',
    options: ['Stack', 'Queue (FIFO - First In First Out)', 'Heap', 'Array'],
    correctAnswer: 1,
    explanation:
      'BFS uses queue (FIFO): oldest added node processed first. This ensures level-by-level traversal. Enqueue neighbors, dequeue and process, repeat.',
  },
  {
    id: 'mc3',
    question: 'Why does BFS guarantee shortest path in unweighted graphs?',
    options: [
      'Random',
      'Explores nodes in order of increasing distance - first discovery is via shortest path',
      'Always faster',
      'Uses sorting',
    ],
    correctAnswer: 1,
    explanation:
      'BFS processes distance d before d+1. When node first discovered, all shorter paths already explored. First discovery = shortest path. DFS no guarantee - may explore long path first.',
  },
  {
    id: 'mc4',
    question: 'When should you use BFS over DFS?',
    options: [
      'All paths needed',
      'Shortest path, level-order traversal, nearest neighbors, minimum steps',
      'Deep exploration',
      'Memory constrained',
    ],
    correctAnswer: 1,
    explanation:
      'Use BFS for: shortest path (unweighted), level-order traversal, finding nodes at specific distance, minimum moves/steps. DFS for: all paths, backtracking, memory-constrained (deep trees).',
  },
  {
    id: 'mc5',
    question: 'What is the space complexity of BFS?',
    options: [
      'O(H) where H is height',
      'O(W) where W is maximum width at any level',
      'O(1)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'BFS uses O(W) space where W is maximum width. Queue holds all nodes at current level. For complete binary tree, last level has N/2 nodes. For graphs, also need O(V) visited set.',
  },
];
