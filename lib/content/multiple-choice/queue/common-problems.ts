/**
 * Multiple choice questions for Common Queue Problems & Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonproblemsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the most common application of queues in algorithms?',
    options: [
      'Sorting',
      'Breadth-First Search (BFS)',
      'Binary search',
      'Finding duplicates',
    ],
    correctAnswer: 1,
    explanation:
      'Queues are essential for BFS, which explores nodes level by level. The queue maintains the order of nodes to visit, ensuring breadth-first traversal.',
  },
  {
    id: 'mc2',
    question:
      'In level-order tree traversal, what determines when to start processing a new level?',
    options: [
      'When the queue is empty',
      'By counting len (queue) at the start of each level',
      'When we see a None node',
      'After processing the root',
    ],
    correctAnswer: 1,
    explanation:
      'We capture len (queue) at the start of each level iteration. This tells us exactly how many nodes are in the current level, allowing us to process them separately.',
  },
  {
    id: 'mc3',
    question:
      'What is the amortized time complexity of dequeue in a two-stack queue implementation?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 0,
    explanation:
      'While individual dequeue operations can be O(n) when transferring elements, the amortized complexity is O(1) because each element is transferred at most once.',
  },
  {
    id: 'mc4',
    question:
      'For the "shortest path in unweighted graph" problem, which algorithm should you use?',
    options: ['DFS', 'BFS with queue', 'Dijkstra', 'Binary search'],
    correctAnswer: 1,
    explanation:
      'BFS with a queue finds the shortest path in unweighted graphs because it explores nodes level by level, guaranteeing the first time you reach a node is via the shortest path.',
  },
  {
    id: 'mc5',
    question:
      'In sliding window maximum, why do we need a deque instead of a priority queue?',
    options: [
      'Deque is faster',
      'Priority queue cannot maintain window order',
      'We need O(1) removal from both ends, priority queue is O(log n)',
      'Deque uses less memory',
    ],
    correctAnswer: 2,
    explanation:
      'We need to efficiently remove elements from both ends: old elements from the front (outside window) and smaller elements from the back. Deque provides O(1) for both, while priority queue would be O(log n).',
  },
];
