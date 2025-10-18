/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a BFS problem?',
    options: [
      'All paths',
      'Shortest path, minimum steps, level-order, nearest, closest',
      'Depth-first',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'BFS keywords: "shortest path", "minimum steps/moves", "level-order", "nearest/closest neighbor", "distance". Contrast: "all paths" → DFS.',
  },
  {
    id: 'mc2',
    question: 'When should you choose BFS over DFS?',
    options: [
      'All paths needed',
      'Shortest path, level-order, nearest nodes, minimum moves, wide shallow trees',
      'Memory constrained',
      'Always',
    ],
    correctAnswer: 1,
    explanation:
      'Choose BFS when: 1) Need shortest path (unweighted), 2) Level-order traversal, 3) Find nearest/closest, 4) Minimum steps, 5) Wide shallow trees (O(W) acceptable). DFS for: all paths, deep narrow trees.',
  },
  {
    id: 'mc3',
    question: 'What should you clarify in a BFS interview?',
    options: [
      'Nothing',
      'Tree vs graph? Weighted? Need path or just distance? Multiple sources?',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Tree or graph (graph needs visited), 2) Weighted edges (use Dijkstra if weighted), 3) Need path or distance only, 4) Single or multiple sources (multi-source BFS), 5) Constraints on graph size.',
  },
  {
    id: 'mc4',
    question: 'What is a common BFS mistake?',
    options: [
      'Using queue',
      'Marking visited after dequeue instead of before enqueue (duplicates in queue), forgetting level tracking',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Mark visited when enqueuing, not dequeuing (else duplicates), 2) Forgetting level tracking pattern, 3) Not handling disconnected components, 4) Using stack instead of queue.',
  },
  {
    id: 'mc5',
    question: 'How should you communicate your BFS solution?',
    options: [
      'Just code',
      'Explain why BFS (shortest path, level-order), queue usage, visited tracking, walk through example, complexity',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Why BFS (shortest path, level-order), 2) Queue FIFO ensures level-by-level, 3) Visited set for graphs, 4) Walk through small example showing levels, 5) Time O(V+E), space O(W).',
  },
];
