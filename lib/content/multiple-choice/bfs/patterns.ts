/**
 * Multiple choice questions for Common BFS Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the multi-source BFS pattern?',
    options: [
      'Multiple BFS runs',
      'Start BFS from multiple sources simultaneously - enqueue all sources first',
      'Random',
      'Cannot do',
    ],
    correctAnswer: 1,
    explanation:
      'Multi-source BFS: enqueue all source nodes initially with distance 0. BFS expands from all simultaneously, processing by distance. Used for: rotten oranges, walls and gates, forest fire spread.',
  },
  {
    id: 'mc2',
    question: 'What is the level-tracking pattern in BFS?',
    options: [
      'Random',
      'Track level/distance: capture queue size, process exactly that many, increment level',
      'No tracking',
      'Use counter',
    ],
    correctAnswer: 1,
    explanation:
      'Level-tracking: before each level, size = len (queue). Process exactly size nodes (current level), add their children (next level). Track level number. Separates levels cleanly.',
  },
  {
    id: 'mc3',
    question: 'What is bidirectional BFS?',
    options: [
      'Two separate BFS',
      'BFS from both source and target simultaneously - meet in middle, faster for large graphs',
      'Random',
      'Cannot do',
    ],
    correctAnswer: 1,
    explanation:
      'Bidirectional BFS runs from source and target simultaneously. Terminate when frontiers meet. Time: O(2×b^(d/2)) vs O(b^d) for single BFS where b is branching, d is distance. Much faster for large graphs.',
  },
  {
    id: 'mc4',
    question: 'How do you handle grid/matrix with BFS?',
    options: [
      'Cannot do',
      'Treat cells as nodes, 4-directional neighbors as edges, mark visited in-place or set',
      'Different algorithm',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Grid BFS: each cell is node. Neighbors: 4 directions [(0,1), (1,0), (0,-1), (-1,0)]. Check bounds, walls, visited. Mark visited in grid or use set. Common for shortest path in maze.',
  },
  {
    id: 'mc5',
    question: 'What is the parent-tracking pattern?',
    options: [
      'Track node parents',
      'Store parent[neighbor] = current during BFS - enables path reconstruction via backtracking',
      'Random',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Parent-tracking: parent[neighbor] = current when discovering. After BFS, reconstruct path: backtrack from target using parent map until reaching source. Reverse for source→target path.',
  },
];
