/**
 * Multiple choice questions for Time and Space Complexity section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of BFS on a tree?',
    options: ['O(log N)', 'O(N) - visit each node once', 'O(N²)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Tree BFS visits each of N nodes exactly once. Each node enqueued and dequeued once. Total O(N) time.',
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
      'Graph BFS: O(V) to visit each vertex with adjacency list, O(E) to explore all edges. With adjacency matrix, neighbor checking is O(V²). Total O(V+E) with list.',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of BFS?',
    options: [
      'O(H) where H is height',
      'O(W) where W is maximum width + O(V) visited for graphs',
      'O(1)',
      'O(N²)',
    ],
    correctAnswer: 1,
    explanation:
      'BFS uses O(W) for queue where W is max width. Complete binary tree: last level has N/2 nodes, so W=O(N). Graphs also need O(V) visited set. Total O(N) or O(V).',
  },
  {
    id: 'mc4',
    question: 'Why is BFS space-inefficient for wide trees?',
    options: [
      'Random',
      'Queue holds entire level - wide trees have many nodes per level O(W)',
      'Always O(1)',
      'No issue',
    ],
    correctAnswer: 1,
    explanation:
      'BFS queue holds all nodes at current level. Wide trees (high branching factor) have many nodes per level. Complete binary tree: last level has N/2 nodes = O(N) space.',
  },
  {
    id: 'mc5',
    question: 'How does BFS space compare to DFS?',
    options: [
      'Same',
      'BFS: O(width) for queue, DFS: O(height) for stack - depends on tree shape',
      'BFS always better',
      'DFS always better',
    ],
    correctAnswer: 1,
    explanation:
      'BFS O(W) width, DFS O(H) height. Balanced tree: both O(N) worst. Deep narrow: DFS worse (H=N). Wide shallow: BFS worse (W=N). Choose based on tree shape.',
  },
];
