/**
 * Multiple choice questions for BFS for Shortest Path section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const shortestpathMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Why does BFS guarantee shortest path?',
    options: [
      'Random',
      'Explores nodes in order of increasing distance from source',
      'Always faster',
      'Uses sorting',
    ],
    correctAnswer: 1,
    explanation:
      'BFS explores distance d before d+1. When node first reached, all shorter paths already explored, so first discovery = shortest path. Queue ensures level-by-level processing.',
  },
  {
    id: 'mc2',
    question: 'How do you reconstruct the shortest path in BFS?',
    options: [
      'Cannot reconstruct',
      'Track parent map during BFS, backtrack from target to source',
      'Store all paths',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'During BFS, store parent[neighbor] = current for each discovered node. After BFS, backtrack from target: path = []; while node: path.append (node); node = parent[node]. Reverse path.',
  },
  {
    id: 'mc3',
    question: 'What if graph has multiple shortest paths?',
    options: [
      'BFS fails',
      'BFS finds one shortest path (first discovered), not necessarily all',
      'Finds all',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'BFS finds one shortest path (whichever explored first). To find all shortest paths, track all parents (list) for each node at same distance, then enumerate all paths.',
  },
  {
    id: 'mc4',
    question: 'Can BFS find shortest path in weighted graphs?',
    options: [
      'Yes always',
      'No - only unweighted or special cases (0-1 BFS). Use Dijkstra for weighted.',
      'Sometimes',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Standard BFS only works for unweighted (all edges weight 1). Weighted graphs need Dijkstra. Exception: 0-1 BFS for graphs with only 0 or 1 weights.',
  },
  {
    id: 'mc5',
    question: 'How do you track distance in BFS?',
    options: [
      'Cannot track',
      'Distance map or tuple in queue: dist[node] = dist[current] + 1 or queue (node, dist)',
      'Count nodes',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Two approaches: 1) Distance map: dist[neighbor] = dist[current] + 1, 2) Queue tuples: queue.append((neighbor, dist+1)). Level-based: distance = level number.',
  },
];
