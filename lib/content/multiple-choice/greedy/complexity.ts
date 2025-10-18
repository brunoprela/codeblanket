/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the typical complexity of greedy algorithms with sorting?',
    options: ['O(N)', 'O(N log N) - sorting dominates', 'O(N²)', 'O(log N)'],
    correctAnswer: 1,
    explanation:
      'Most greedy algorithms require O(N log N) sorting followed by O(N) greedy processing, giving O(N log N) total complexity.',
  },
  {
    id: 'mc2',
    question: 'What is the space complexity of most greedy algorithms?',
    options: [
      'O(N log N)',
      'O(1) to O(N) - often in-place or linear auxiliary space',
      'O(N²)',
      'O(2^N)',
    ],
    correctAnswer: 1,
    explanation:
      "Greedy algorithms typically use O(1) space (in-place like jump game) or O(N) for sorting/auxiliary structures. Much better than DP's O(N²) space.",
  },
  {
    id: 'mc3',
    question: 'How can you avoid sorting to optimize greedy algorithms?',
    options: [
      'Cannot avoid sorting',
      'Use heap/priority queue for dynamic best-choice selection',
      'Use arrays only',
      'Random selection',
    ],
    correctAnswer: 1,
    explanation:
      'When greedy choice changes dynamically, use priority queue to maintain best choice in O(log N) instead of O(N log N) re-sorting. Useful for streaming/online problems.',
  },
  {
    id: 'mc4',
    question: 'What is the complexity of jump game with greedy?',
    options: ['O(N log N)', 'O(N) single pass, O(1) space', 'O(N²)', 'O(2^N)'],
    correctAnswer: 1,
    explanation:
      'Jump game: one pass tracking farthest reachable position. O(N) time, O(1) space. No sorting needed. This is optimal.',
  },
  {
    id: 'mc5',
    question: 'How does greedy complexity compare to DP for same problems?',
    options: [
      'Always the same',
      'Greedy usually faster: O(N log N) vs DP O(N²), but works on fewer problems',
      'DP is always faster',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "When greedy works, it's typically O(N log N) vs DP's O(N²). But greedy only works when greedy choice property holds - DP is more general.",
  },
];
