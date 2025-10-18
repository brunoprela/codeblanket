/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal using Fenwick Tree in an interview?',
    options: [
      'Sorting',
      'Prefix sum, range sum, cumulative frequency, inversions - with updates',
      'Shortest path',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree signals: "prefix sum", "range sum", "cumulative frequency", "count inversions", "dynamic array queries". Key: need both queries AND updates efficiently.',
  },
  {
    id: 'mc2',
    question: 'What should you clarify in a Fenwick Tree interview?',
    options: [
      'Nothing',
      'Operation type (sum/XOR?), update frequency, can I use 1-indexing?',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) What operation (if min/max, need Segment Tree), 2) How frequent are updates (if none, prefix sum array faster), 3) Can use 1-indexing (standard for Fenwick), 4) Value range (coordinate compression?).',
  },
  {
    id: 'mc3',
    question: 'What is a common mistake when implementing Fenwick Tree?',
    options: [
      'Using bit manipulation',
      'Forgetting 1-based indexing, off-by-one in range queries',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Using 0-based indexing (breaks algorithm), 2) Range query: prefix_sum(R) - prefix_sum(L-1), forgetting L-1, 3) Update vs set confusion (Fenwick adds delta, not sets value).',
  },
  {
    id: 'mc4',
    question: 'How should you communicate your Fenwick Tree solution?',
    options: [
      'Just code',
      'Explain why Fenwick (O(log N) for sum+update), mention bit manipulation briefly, walk through example, complexity',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Why Fenwick over alternatives (simpler than Segment Tree for sums), 2) Briefly mention bit manipulation concept, 3) Walk through one update and query, 4) Time O(log N), space O(N), 5) Code with comments.',
  },
  {
    id: 'mc5',
    question: 'What makes Fenwick Tree interview-friendly?',
    options: [
      'Random',
      'Short code (~20 lines), easy to memorize, fewer bugs than Segment Tree',
      'Complex',
      'Slow',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree is interview-friendly: 1) Only ~20 lines total, 2) 2 simple functions, easy to memorize, 3) Less error-prone than Segment Tree (50+ lines), 4) Same O(log N) performance.',
  },
];
