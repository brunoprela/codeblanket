/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What data structure does BFS template use?',
    options: [
      'Stack',
      'Queue (deque) for level-by-level exploration',
      'Heap',
      'Array',
    ],
    correctAnswer: 1,
    explanation:
      'BFS template uses queue (collections.deque). Add start, while queue: pop, process, add unvisited neighbors. Queue ensures level-by-level exploration.',
  },
  {
    id: 'mc2',
    question:
      'What is the key difference between iterative and recursive DFS templates?',
    options: [
      'Speed',
      'Iterative uses explicit stack, recursive uses call stack',
      'Complexity',
      'They are the same',
    ],
    correctAnswer: 1,
    explanation:
      'DFS: recursive uses call stack (cleaner code), iterative uses explicit stack (more control, avoids stack overflow). Both explore depth-first, same complexity.',
  },
  {
    id: 'mc3',
    question:
      'In Union-Find template, what do path compression and union by rank do?',
    options: [
      'Nothing',
      'Flatten trees during find and attach smaller under larger for O(α(N)) amortized',
      'Sort elements',
      'Random optimization',
    ],
    correctAnswer: 1,
    explanation:
      'Path compression: flatten tree during find (make nodes point to root). Union by rank: attach smaller tree under larger. Together give O(α(N)) ≈ O(1) per operation.',
  },
  {
    id: 'mc4',
    question: 'What is common to all graph traversal templates?',
    options: [
      'Sorting',
      'Visited set to track processed nodes and prevent cycles',
      'Heap usage',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'All graph templates use visited set to: 1) prevent infinite loops in cycles, 2) ensure each node processed once, 3) achieve O(V+E) complexity.',
  },
  {
    id: 'mc5',
    question: 'When would you modify the basic BFS template?',
    options: [
      'Never',
      'Track distance/level, find shortest path, level-order specific logic',
      'Always',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Modify BFS for: tracking distance (distance array), shortest path (parent pointers), level-order processing (track level), multi-source BFS (start with multiple nodes).',
  },
];
