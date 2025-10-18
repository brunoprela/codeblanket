/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a DFS problem?',
    options: [
      'Shortest path',
      'Explore all paths, find path, detect cycle, traverse tree, backtracking',
      'Level order',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'DFS keywords: "all paths" (not shortest), "traverse tree", "detect cycle", "topological sort", "connected components", "backtracking". Contrast: "shortest path" → BFS.',
  },
  {
    id: 'mc2',
    question: 'When should you choose DFS over BFS?',
    options: [
      'Need shortest path',
      'Exploring all paths, memory constrained (deep tree), backtracking, topological sort',
      'Level-order traversal',
      'Always',
    ],
    correctAnswer: 1,
    explanation:
      'Choose DFS when: 1) Need all paths (not just shortest), 2) Memory limited (O(H) better for deep narrow trees), 3) Backtracking problems, 4) Topological sort, 5) Tree traversal (natural recursion).',
  },
  {
    id: 'mc3',
    question: 'What should you clarify in a DFS interview?',
    options: [
      'Nothing',
      'Tree vs graph? Cycles possible? Need all paths or one? Can modify input?',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Tree or graph (graph needs visited set), 2) Directed/undirected, cycles? 3) Find one path or all? 4) Can modify input (mark visited in-place)? 5) Connectivity (one component or multiple)?',
  },
  {
    id: 'mc4',
    question: 'What is a common DFS mistake?',
    options: [
      'Using recursion',
      'Forgetting visited set for graphs, not backtracking properly, infinite recursion',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Forgetting visited set (infinite loop on cycles), 2) Not backtracking (not undoing path modifications), 3) No base case (infinite recursion), 4) Modifying list while iterating.',
  },
  {
    id: 'mc5',
    question: 'How should you communicate your DFS solution?',
    options: [
      'Just code',
      'Explain recursive structure, base case, why DFS over BFS, walk through example, mention complexity',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Why DFS (all paths, tree, etc), 2) Explain recursive structure (what info passed/returned), 3) Base cases clearly, 4) Walk through small example, 5) Time O(V+E) / space O(H), 6) Backtracking if applicable.',
  },
];
