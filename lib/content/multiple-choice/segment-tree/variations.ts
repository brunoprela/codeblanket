/**
 * Multiple choice questions for Segment Tree Variations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const variationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What variation do you use for range minimum query?',
    options: [
      'Same as sum',
      'Change combine to min(left, right) instead of left + right',
      'Different tree structure',
      'Cannot do RMQ',
    ],
    correctAnswer: 1,
    explanation:
      'RMQ: just change combine function from sum (left+right) to min (min(left,right)). Identity changes from 0 to INF. Same structure, different operation.',
  },
  {
    id: 'mc2',
    question: 'How does lazy propagation work for range updates?',
    options: [
      'Updates immediately',
      'Store pending updates in lazy array, push down only when needed',
      'Rebuilds tree',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Lazy propagation: maintain lazy[node] array for pending updates. When updating range, mark affected nodes as lazy. Push updates down only when querying/updating those nodes. Reduces range update to O(log N).',
  },
  {
    id: 'mc3',
    question: 'What operations can Segment Tree handle?',
    options: [
      'Only sum',
      'Any associative operation: sum, min, max, GCD, XOR, etc.',
      'Only min/max',
      'None',
    ],
    correctAnswer: 1,
    explanation:
      "Segment Tree works with any associative operation where order of combining doesn't matter: sum, product, min, max, GCD, LCM, XOR, OR, AND. Just change the combine function.",
  },
  {
    id: 'mc4',
    question: '2D Segment Tree is used for what?',
    options: [
      'Sorting',
      'Range queries on 2D matrix (rectangle sum, min, etc.)',
      'Graph traversal',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      '2D Segment Tree handles 2D range queries like rectangle sum in O(log²N) time. Build tree of trees: outer for rows, inner for columns. Updates and queries work in both dimensions.',
  },
  {
    id: 'mc5',
    question: 'When would you use Persistent Segment Tree?',
    options: [
      'Random',
      'Need to query previous versions of array after updates (version control)',
      'Always',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Persistent Segment Tree maintains all versions after updates by creating new nodes instead of modifying. Query any historical version. Used in time-travel queries, undo/redo, or range queries at specific timestamps.',
  },
];
