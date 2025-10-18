/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of Segment Tree operations?',
    options: [
      'All O(N)',
      'Build: O(N), Query: O(log N), Update: O(log N)',
      'All O(log N)',
      'Build: O(log N), Query: O(N)',
    ],
    correctAnswer: 1,
    explanation:
      'Build: O(N) visits each node once. Query: O(log N) traverses tree height. Point Update: O(log N) updates path to root. Range Update with lazy prop: O(log N).',
  },
  {
    id: 'mc2',
    question: 'What is the space complexity of Segment Tree?',
    options: [
      'O(log N)',
      'O(N) - specifically 4N for array representation',
      'O(N²)',
      'O(1)',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree needs O(N) space. Array representation uses 4N to handle all cases safely (complete binary tree with N leaves needs at most 2N-1 nodes, 4N is safe upper bound).',
  },
  {
    id: 'mc3',
    question: 'Why is range query O(log N)?',
    options: [
      'Random',
      'At most O(log N) nodes visited per level, and tree height is O(log N)',
      'Always fast',
      'Uses binary search',
    ],
    correctAnswer: 1,
    explanation:
      'Range query visits at most 2 nodes per level (one for each subtree boundary). Tree height is O(log N), so total nodes visited is O(log N).',
  },
  {
    id: 'mc4',
    question: 'How does lazy propagation reduce range update complexity?',
    options: [
      'Makes it slower',
      'Defers updates - mark O(log N) nodes instead of updating O(N) affected leaves',
      'Random',
      'No difference',
    ],
    correctAnswer: 1,
    explanation:
      'Without lazy prop: range update touches O(N) leaves. With lazy prop: mark O(log N) ancestor nodes, push updates only when needed. Reduces from O(N) to O(log N).',
  },
  {
    id: 'mc5',
    question: 'What makes Segment Tree efficient?',
    options: [
      'Random',
      'Precomputed intervals at all levels enable O(log N) range queries by combining logarithmic nodes',
      'Always O(1)',
      'Uses sorting',
    ],
    correctAnswer: 1,
    explanation:
      'Segment Tree precomputes all interval combinations. Any range can be decomposed into O(log N) precomputed intervals. Combining these is fast (single operation per interval).',
  },
];
