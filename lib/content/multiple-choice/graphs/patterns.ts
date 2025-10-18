/**
 * Multiple choice questions for Common Graph Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const patternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do you find connected components in an undirected graph?',
    options: [
      'Sort nodes',
      'Run BFS/DFS from each unvisited node, each traversal finds one component',
      'Use heap',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'Connected components: iterate through nodes, for each unvisited node, run BFS/DFS (marks all reachable nodes). Each traversal finds one component. Count of traversals = number of components.',
  },
  {
    id: 'mc2',
    question: 'How do you detect a cycle in a directed graph?',
    options: [
      'BFS only',
      'DFS with recursion stack - if we visit a node already in current path, cycle exists',
      'Sort',
      'Cannot detect',
    ],
    correctAnswer: 1,
    explanation:
      'Directed cycle: DFS with recursion stack. Track nodes in current path. If we reach a node already in path (not just visited), found a cycle back to ancestor.',
  },
  {
    id: 'mc3',
    question:
      'What makes topological sort possible for DAGs but not cyclic graphs?',
    options: [
      'Speed',
      'DAGs have no cycles - can order linearly where all edges go forward',
      'Random',
      'Size',
    ],
    correctAnswer: 1,
    explanation:
      'Topological sort orders nodes so all edges go from earlier to later. Cycles make this impossible (circular dependency). DAGs (acyclic) guarantee such ordering exists.',
  },
  {
    id: 'mc4',
    question: 'How do you check if a graph is bipartite?',
    options: [
      'Count nodes',
      'Try 2-coloring with BFS/DFS - if adjacent nodes get same color, not bipartite',
      'Sort edges',
      'Random testing',
    ],
    correctAnswer: 1,
    explanation:
      'Bipartite: attempt to 2-color graph. BFS/DFS alternating colors. If we try to color a node but it already has different color, graph has odd cycle and is not bipartite.',
  },
  {
    id: 'mc5',
    question: 'What is Union-Find used for in graph problems?',
    options: [
      'Sorting',
      'Efficiently tracking connected components, detecting cycles in undirected graphs',
      'Shortest path',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Union-Find (Disjoint Set): efficiently merge components (union) and check if nodes connected (find). O(α(N)) amortized per operation. Used for Kruskal MST, cycle detection.',
  },
];
