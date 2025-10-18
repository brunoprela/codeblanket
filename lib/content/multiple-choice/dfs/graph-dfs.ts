/**
 * Multiple choice questions for DFS on Graphs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const graphdfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Why do you need a visited set for graph DFS?',
    options: [
      'Optimization',
      'Prevent infinite loops from cycles - mark nodes as visited',
      'Random requirement',
      'Faster processing',
    ],
    correctAnswer: 1,
    explanation:
      'Graphs can have cycles (unlike trees). Without visited tracking, DFS would loop infinitely A→B→A→B... Visited set prevents revisiting nodes, ensuring O(V+E) time.',
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
      'DFS visits each vertex once O(V) and explores each edge once O(E). Total: O(V + E). With adjacency list, checking neighbors is O(degree), summing to O(E) across all vertices.',
  },
  {
    id: 'mc3',
    question: 'How does DFS detect cycles in a directed graph?',
    options: [
      'Cannot detect',
      'Track recursion stack - if edge leads to node in stack (visiting state), cycle exists',
      'Count edges',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'For directed graph: use 3 states (unvisited, visiting in current path, visited done). Cycle exists if back edge to "visiting" node (ancestor in current recursion path). Visiting state tracks current DFS path.',
  },
  {
    id: 'mc4',
    question: 'What is topological sort and how does DFS compute it?',
    options: [
      'Random sorting',
      'Linear ordering of DAG where u→v means u before v. DFS: postorder (finish time), reverse gives topo order',
      'BFS only',
      'Cannot do',
    ],
    correctAnswer: 1,
    explanation:
      'Topological sort orders DAG respecting edges. DFS: run DFS, add vertex to stack after exploring all neighbors (postorder/finish time). Reverse stack = topological order. Only works on DAGs.',
  },
  {
    id: 'mc5',
    question: 'What is the space complexity of DFS on a graph?',
    options: [
      'O(1)',
      'O(V) for visited set + O(H) for recursion stack, worst case H=V',
      'O(E)',
      'O(V²)',
    ],
    correctAnswer: 1,
    explanation:
      'Space: O(V) for visited set tracking all vertices, O(H) for recursion call stack where H is DFS depth. Worst case linear graph H=V. Total O(V).',
  },
];
